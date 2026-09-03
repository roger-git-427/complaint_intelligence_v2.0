"""Complaint narrative index: sentence embeddings when available, else TF-IDF."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import INDEX_PATH, MODELS_DIR, SILVER_PATH

RECORD_FIELDS = (
    "complaint_id",
    "company",
    "product",
    "issue",
    "sub_issue",
    "state",
    "date_received",
    "narrative",
)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = 64


def _load_narratives() -> pd.DataFrame:
    if not SILVER_PATH.exists():
        raise FileNotFoundError(f"Missing {SILVER_PATH}. Run: python etl/transform.py")
    frame = pd.read_parquet(SILVER_PATH, columns=list(RECORD_FIELDS))
    text = frame["narrative"].fillna("").astype(str).str.strip()
    frame = frame.loc[text.str.len() >= 40].copy()
    frame["narrative"] = text.loc[frame.index]
    for column in RECORD_FIELDS:
        if column == "date_received":
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")
        frame[column] = frame[column].astype("string").fillna("")
    return frame.reset_index(drop=True)


def _try_sentence_model():
    if os.environ.get("FORCE_TFIDF_INDEX", "").lower() in {"1", "true", "yes"}:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBED_MODEL)
    except Exception as exc:
        print(f"sentence-transformers unavailable ({exc}); using TF-IDF index")
        return None


def build_index() -> Path:
    frame = _load_narratives()
    texts = frame["narrative"].tolist()
    records = frame[list(RECORD_FIELDS)].to_dict(orient="records")
    model = _try_sentence_model()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if model is not None:
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        payload = {
            "backend": "embeddings",
            "model_name": EMBED_MODEL,
            "embeddings": np.asarray(embeddings, dtype=np.float32),
            "records": records,
        }
        print(f"RAG index (embeddings/{EMBED_MODEL}): {len(records)} rows -> {INDEX_PATH}")
    else:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50_000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        payload = {
            "backend": "tfidf",
            "vectorizer": vectorizer,
            "matrix": matrix,
            "records": records,
        }
        print(f"RAG index (tfidf): {len(records)} rows -> {INDEX_PATH}")

    joblib.dump(payload, INDEX_PATH)
    return INDEX_PATH


class ComplaintIndex:
    def __init__(self, path: Path = INDEX_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run: python ml/train.py")
        payload = joblib.load(path)
        self.backend = payload.get("backend", "tfidf")
        self.records = payload["records"]
        self.model_name = payload.get("model_name")
        self._embedder = None
        if self.backend == "embeddings":
            self.embeddings = payload["embeddings"]
        else:
            self.vectorizer = payload["vectorizer"]
            self.matrix = payload["matrix"]

    def _encode_query(self, query: str) -> np.ndarray:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.model_name or EMBED_MODEL)
        vector = self._embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vector, dtype=np.float32)

    def search(self, query: str, k: int = 5) -> list[dict]:
        cleaned = (query or "").strip()
        if not cleaned:
            return []
        k = max(1, min(int(k), 20))
        if self.backend == "embeddings":
            query_vec = self._encode_query(cleaned)
            scores = (self.embeddings @ query_vec.T).ravel()
        else:
            scores = cosine_similarity(self.vectorizer.transform([cleaned]), self.matrix).ravel()
        hits: list[dict] = []
        for idx in scores.argsort()[::-1][:k]:
            record = dict(self.records[int(idx)])
            narrative = str(record.get("narrative") or "")
            record["score"] = round(float(scores[idx]), 4)
            record["snippet"] = narrative[:400] + ("..." if len(narrative) > 400 else "")
            record["backend"] = self.backend
            hits.append(record)
        return hits
