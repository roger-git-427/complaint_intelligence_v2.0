"""FastAPI: /predict, /predict/timely, /search, /sql, /chat."""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from etl.run_queries import reset_cache
from paths import CLASSIFIER_PATH, METRICS_PATH, TIMELY_CLASSIFIER_PATH, TIMELY_METRICS_PATH
from rag.chat import answer as chat_answer
from rag.retriever import ComplaintIndex
from rag.text_to_sql import run_question

STATIC_DIR = Path(__file__).resolve().parent / "static"
TRAIN_HINT = "Run: python ml/train.py"
STATE: dict = {
    "classifier": None,
    "timely_classifier": None,
    "index": None,
    "metrics": None,
    "timely_metrics": None,
}


class PredictRequest(BaseModel):
    narrative: str = Field(min_length=10)


class SearchRequest(BaseModel):
    query: str = Field(min_length=3)
    k: int = Field(default=5, ge=1, le=20)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=5, ge=1, le=20)


def _require(key: str):
    obj = STATE[key]
    if obj is None:
        raise HTTPException(503, TRAIN_HINT)
    return obj


@asynccontextmanager
async def lifespan(_app: FastAPI):
    reset_cache()
    if CLASSIFIER_PATH.exists():
        STATE["classifier"] = joblib.load(CLASSIFIER_PATH)
    if TIMELY_CLASSIFIER_PATH.exists():
        STATE["timely_classifier"] = joblib.load(TIMELY_CLASSIFIER_PATH)
    if METRICS_PATH.exists():
        STATE["metrics"] = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if TIMELY_METRICS_PATH.exists():
        STATE["timely_metrics"] = json.loads(TIMELY_METRICS_PATH.read_text(encoding="utf-8"))
    try:
        STATE["index"] = ComplaintIndex()
    except FileNotFoundError:
        STATE["index"] = None
    yield
    reset_cache()


app = FastAPI(title="Consumer Complaint Intelligence", lifespan=lifespan)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return {"docs": "/docs"}
    return FileResponse(page)


@app.get("/health")
def health():
    index = STATE["index"]
    return {
        "ok": True,
        "classifier": STATE["classifier"] is not None,
        "timely_classifier": STATE["timely_classifier"] is not None,
        "index": index is not None,
        "index_backend": getattr(index, "backend", None),
        "labels": (STATE["metrics"] or {}).get("labels", []),
    }


@app.get("/metrics")
def metrics():
    return {
        "product": _require("metrics"),
        "timely": STATE["timely_metrics"],
    }


@app.post("/predict")
def predict(body: PredictRequest):
    model = _require("classifier")
    proba = model.predict_proba([body.narrative])[0]
    ranked = sorted(zip(model.classes_, proba, strict=True), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    return {
        "product": top_label,
        "confidence": round(float(top_score), 4),
        "alternatives": [
            {"product": label, "confidence": round(float(score), 4)} for label, score in ranked[:5]
        ],
    }


@app.post("/predict/timely")
def predict_timely(body: PredictRequest):
    model = _require("timely_classifier")
    proba = float(model.predict_proba([body.narrative])[0][1])
    return {
        "timely_response": proba >= 0.5,
        "confidence": round(proba if proba >= 0.5 else 1.0 - proba, 4),
        "probability_timely": round(proba, 4),
    }


@app.post("/search")
def search(body: SearchRequest):
    return {"hits": _require("index").search(body.query, k=body.k)}


@app.post("/sql")
def sql(body: QuestionRequest):
    result = run_question(body.question)
    return {"sql": result.sql, "source": result.source, "rows": result.rows, "error": result.error}


@app.post("/chat")
def chat(body: QuestionRequest):
    return chat_answer(body.question, _require("index"), k=body.k)
