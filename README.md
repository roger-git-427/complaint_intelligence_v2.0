# Consumer Complaint Intelligence Platform

End-to-end **lakehouse** on real [CFPB consumer complaints](https://www.consumerfinance.gov/data-research/consumer-complaints/): bronze → silver → gold transforms, star-schema SQL on DuckDB, MLflow-tracked classifiers, MiniLM retrieval, and a FastAPI demo for prediction, search, and guarded text-to-SQL.

## Features

| Layer | Implementation |
|---|---|
| Ingest | CFPB API → bulk zip → Hugging Face fallback |
| Lakehouse | `data/bronze` → `data/silver` → `data/gold` parquet |
| Quality | Uniqueness, nulls, FK checks (`etl/quality.py`) |
| Analytics | Cached DuckDB over gold + window-function SQL |
| ML | Product + timely-response classifiers (TF-IDF + logistic regression), time split, MLflow |
| Retrieval | `sentence-transformers` (`all-MiniLM-L6-v2`); TF-IDF fallback |
| API | `/predict`, `/predict/timely`, `/search`, `/sql`, `/chat` + demo UI |

## Requirements

- Python **3.11+** (CI uses 3.12)
- ~2 GB disk for a small CFPB extract + MiniLM download on first train

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements-dev.txt

python etl/download_cfpb.py --source auto --years 2024 2025 --max-records 8000
python etl/transform.py
python etl/quality.py
python etl/run_queries.py
python ml/train.py

python -m uvicorn api.main:app --host 127.0.0.1 --port 8080
```

Then open http://127.0.0.1:8080 (UI) or http://127.0.0.1:8080/docs (OpenAPI).

First train downloads MiniLM once. To force TF-IDF retrieval instead:

```bash
# Windows PowerShell
$env:FORCE_TFIDF_INDEX=1
python ml/train.py

# macOS / Linux
# FORCE_TFIDF_INDEX=1 python ml/train.py
```

**Docker (API only):** train locally so `models/` and `data/gold` exist, then `docker compose up --build`.

Demo curls: [docs/demo.md](docs/demo.md) · Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)

## Model metrics

Held-out results from a ~8k narrative extract (2024), time-based 80/20 split. Prefer **macro F1** over accuracy — credit reporting dominates the mix.

### Product classifier

| Metric | Value |
|---|---|
| Accuracy | 0.80 |
| Macro F1 | 0.68 |
| Weighted F1 | 0.80 |

| Product | F1 | Support |
|---|---:|---:|
| Mortgage | 0.89 | 129 |
| Credit reporting | 0.86 | 725 |
| Checking / savings | 0.79 | 215 |
| Credit card | 0.79 | 188 |
| Student loan | 0.70 | 9 |
| Debt collection | 0.70 | 257 |
| Vehicle loan / lease | 0.44 | 39 |
| Money transfer | 0.29 | 27 |

Confusion matrix: [`models/confusion_matrix.png`](models/confusion_matrix.png)

### Timely-response classifier

| Metric | Value |
|---|---|
| Accuracy | 0.99 |
| Macro F1 | 0.50 |
| ROC AUC | 0.81 |

Accuracy is inflated by imbalance (most responses are timely). Prefer ROC AUC / macro F1.

Full JSON: `models/metrics.json`, `models/metrics_timely.json`.

### Limitations

- Rare classes stay weak on an 8k sample.
- Timely prediction from narrative alone is a weak signal.
- Embeddings are local MiniLM, not a managed vector DB.

## Repo layout

```
etl/          download, transform, quality
sql/          star schema + analytical queries
ml/           trainers (MLflow)
rag/          retrieval, text-to-SQL, chat routing
api/          FastAPI + static UI
tests/        pytest + CI
docs/         demo walkthrough
models/       committed metrics + confusion matrix (binaries gitignored)
```

## Tests & CI

```bash
pip install -r requirements-dev.txt
pytest
ruff check etl ml rag api paths.py tests
```

GitHub Actions runs lint + pytest on push/PR to `main` or `master`.

## Summary

End-to-end consumer-complaint intelligence lakehouse on public CFPB data: bronze/silver/gold transforms with data-quality checks, star-schema analytics on DuckDB, MLflow-tracked product and timely-response classifiers, MiniLM embedding retrieval, and FastAPI for prediction, search, and guarded text-to-SQL.

## Data & privacy

Complaint narratives come from the CFPB public database. Respect [CFPB data use](https://www.consumerfinance.gov/complaint/data-use/) guidance. Raw extracts, `.joblib` models, and `.env` are gitignored — only metrics JSON and the confusion matrix are committed under `models/`.

## License

MIT — see [LICENSE](LICENSE).
