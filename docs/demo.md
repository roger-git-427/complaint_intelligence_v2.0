# Demo walkthrough

Assumes you already ran download → transform → train and the API is up on port **8080**.

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8080
```

## Health

```bash
curl http://127.0.0.1:8080/health
```

Expect `classifier: true`, `timely_classifier: true`, and `index_backend` of `embeddings` (or `tfidf` if forced).

## Predict product

```bash
curl -s http://127.0.0.1:8080/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"narrative\": \"Experian will not remove late payments that belong to an identity thief even after I sent police reports.\"}"
```

## Predict timely response

```bash
curl -s http://127.0.0.1:8080/predict/timely ^
  -H "Content-Type: application/json" ^
  -d "{\"narrative\": \"The bank answered within two days and closed my case with a refund.\"}"
```

## Similar complaints (embeddings)

```bash
curl -s http://127.0.0.1:8080/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"Unauthorized accounts appeared on my credit report after a data breach\", \"k\": 3}"
```

Hits include `backend` (`embeddings` or `tfidf`).

## Warehouse question (text-to-SQL)

```bash
curl -s http://127.0.0.1:8080/sql ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"Which companies have the most credit card complaints in Texas?\"}"
```

## Chat router

```bash
curl -s http://127.0.0.1:8080/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"Which companies have the most credit card complaints in Texas?\"}"
```

## Data quality

```bash
python etl/quality.py
```

Expect all checks `PASS` and `models/quality_report.json`.
