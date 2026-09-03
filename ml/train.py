"""Train product + timely classifiers; write metrics plots; rebuild RAG index."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (
    CLASSIFIER_PATH,
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    MLFLOW_DB,
    MODELS_DIR,
    SILVER_PATH,
    TIMELY_CLASSIFIER_PATH,
    TIMELY_METRICS_PATH,
)

MIN_NARRATIVE_CHARS = 40
MIN_CLASS_COUNT = 40
TEST_FRACTION = 0.2


def load_training_frame() -> pd.DataFrame:
    if not SILVER_PATH.exists():
        raise FileNotFoundError(f"Missing {SILVER_PATH}. Run: python etl/transform.py")
    frame = pd.read_parquet(SILVER_PATH)
    text = frame["narrative"].fillna("").astype(str).str.strip()
    frame = frame.loc[text.str.len() >= MIN_NARRATIVE_CHARS].copy()
    frame["narrative"] = text.loc[frame.index]
    frame["product"] = frame["product"].astype(str)
    counts = frame["product"].value_counts()
    dropped = counts[counts < MIN_CLASS_COUNT]
    if len(dropped):
        print("Dropping rare products:")
        for name, count in dropped.items():
            print(f"  {count:4d}  {name}")
        frame = frame.loc[frame["product"].isin(counts[counts >= MIN_CLASS_COUNT].index)]
    frame["date_received"] = pd.to_datetime(frame["date_received"], utc=True)
    return frame.sort_values("date_received").reset_index(drop=True)


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=40_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=400, class_weight="balanced", C=2.0, solver="lbfgs"),
            ),
        ]
    )


def time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = max(1, min(int(len(frame) * (1.0 - TEST_FRACTION)), len(frame) - 1))
    return frame.iloc[:cut], frame.iloc[cut:]


def save_confusion_matrix(y_true, y_pred, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=True,
        ax=ax,
    )
    ax.set_title("Product classifier — held-out time split")
    fig.tight_layout()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=140)
    plt.close(fig)
    tqdm.write(f"Wrote {CONFUSION_MATRIX_PATH}")


def log_mlflow(metrics: dict, pipeline: Pipeline, experiment: str, run_name: str) -> None:
    try:
        import mlflow
        import mlflow.sklearn

        uri = "sqlite:///" + MLFLOW_DB.resolve().as_posix()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "model": "TfidfVectorizer+LogisticRegression",
                    "min_class_count": MIN_CLASS_COUNT,
                    "test_fraction": TEST_FRACTION,
                    "max_features": 40_000,
                }
            )
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key != "classification_report":
                    mlflow.log_metric(key, value)
            mlflow.sklearn.log_model(pipeline, "model")
            if CONFUSION_MATRIX_PATH.exists() and experiment.endswith("product-classifier"):
                mlflow.log_artifact(str(CONFUSION_MATRIX_PATH))
        tqdm.write(f"MLflow run logged at {uri} ({experiment})")
    except Exception as exc:
        tqdm.write(f"MLflow skipped ({exc})")


def train_product(frame: pd.DataFrame) -> tuple[Pipeline, dict]:
    train, test = time_split(frame)
    print(
        f"Product train {len(train)} ({train['date_received'].min().date()} .. {train['date_received'].max().date()})\n"
        f"Product test  {len(test)} ({test['date_received'].min().date()} .. {test['date_received'].max().date()})"
    )
    pipeline = make_pipeline()
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    x_train = vectorizer.fit_transform(train["narrative"])
    classifier.fit(x_train, train["product"])
    predicted = classifier.predict(vectorizer.transform(test["narrative"]))
    labels = sorted(frame["product"].unique().tolist())
    report = classification_report(test["product"], predicted, output_dict=True, zero_division=0)
    save_confusion_matrix(test["product"], predicted, labels)

    per_class = {
        label: {
            "precision": round(report[label]["precision"], 4),
            "recall": round(report[label]["recall"], 4),
            "f1": round(report[label]["f1-score"], 4),
            "support": int(report[label]["support"]),
        }
        for label in labels
        if label in report
    }
    metrics = {
        "task": "product",
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(f1_score(test["product"], predicted, average="macro", zero_division=0), 4),
        "weighted_f1": round(
            f1_score(test["product"], predicted, average="weighted", zero_division=0), 4
        ),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": str(train["date_received"].min().date()),
        "train_end": str(train["date_received"].max().date()),
        "test_start": str(test["date_received"].min().date()),
        "test_end": str(test["date_received"].max().date()),
        "labels": labels,
        "label_counts": {k: int(v) for k, v in frame["product"].value_counts().items()},
        "per_class": per_class,
        "classification_report": report,
        "limitations": [
            "Credit reporting dominates the label mix; prefer macro F1 over accuracy.",
            "Rare products (money transfer, vehicle) remain weak with this sample size.",
            "TF-IDF + logistic regression is a strong baseline, not a production ensemble.",
        ],
    }
    return pipeline, metrics


def train_timely(frame: pd.DataFrame) -> tuple[Pipeline, dict] | tuple[None, None]:
    timely = frame.dropna(subset=["timely_response"]).copy()
    timely["timely_response"] = timely["timely_response"].astype(bool)
    if timely["timely_response"].nunique() < 2:
        tqdm.write("Timely model skipped: need both Yes and No labels")
        return None, None
    train, test = time_split(timely)
    pipeline = make_pipeline()
    pipeline.fit(train["narrative"], train["timely_response"].astype(int))
    proba = pipeline.predict_proba(test["narrative"])[:, 1]
    predicted = (proba >= 0.5).astype(int)
    y_true = test["timely_response"].astype(int)
    report = classification_report(y_true, predicted, output_dict=True, zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auc = None
    metrics = {
        "task": "timely_response",
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(f1_score(y_true, predicted, average="macro", zero_division=0), 4),
        "roc_auc": None if auc is None else round(auc, 4),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "positive_rate_train": round(float(train["timely_response"].mean()), 4),
        "positive_rate_test": round(float(test["timely_response"].mean()), 4),
        "classification_report": report,
        "limitations": [
            "Timely response is often highly imbalanced toward Yes in CFPB extracts.",
            "Narrative text alone is a weak signal for company response latency.",
        ],
    }
    return pipeline, metrics


def main() -> int:
    frame = load_training_frame()
    stages = tqdm(total=5, desc="Training", unit="step", dynamic_ncols=True)

    stages.set_postfix_str("product classifier")
    product_model, product_metrics = train_product(frame)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(product_model, CLASSIFIER_PATH)
    METRICS_PATH.write_text(json.dumps(product_metrics, indent=2, default=str), encoding="utf-8")
    tqdm.write(
        f"product accuracy={product_metrics['accuracy']}  macro_f1={product_metrics['macro_f1']}"
    )
    log_mlflow(product_metrics, product_model, "cfpb-product-classifier", "tfidf-logreg-product")
    stages.update(1)

    stages.set_postfix_str("timely classifier")
    timely_model, timely_metrics = train_timely(frame)
    if timely_model is not None and timely_metrics is not None:
        joblib.dump(timely_model, TIMELY_CLASSIFIER_PATH)
        TIMELY_METRICS_PATH.write_text(
            json.dumps(timely_metrics, indent=2, default=str), encoding="utf-8"
        )
        tqdm.write(
            f"timely accuracy={timely_metrics['accuracy']}  macro_f1={timely_metrics['macro_f1']}  "
            f"auc={timely_metrics['roc_auc']}"
        )
        log_mlflow(timely_metrics, timely_model, "cfpb-timely-classifier", "tfidf-logreg-timely")
    stages.update(1)

    stages.set_postfix_str("data quality checks")
    from etl.quality import run_checks

    quality = run_checks()
    tqdm.write(f"quality passed={quality['passed']} ({len(quality['checks'])} checks)")
    stages.update(1)

    stages.set_postfix_str("building RAG index")
    from rag.retriever import build_index

    build_index()
    stages.update(1)

    stages.set_postfix_str("done")
    stages.update(1)
    stages.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
