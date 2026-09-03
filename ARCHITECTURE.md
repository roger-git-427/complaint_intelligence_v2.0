# Architecture

## Stack

```mermaid
flowchart LR
  A[CFPB / Hugging Face extract] --> B[data/raw CSV]
  B --> C[bronze parquet]
  C --> D[silver parquet]
  D --> E[gold star schema parquet]
  E --> F[Cached DuckDB SQL]
  D --> G[Product + timely classifiers]
  D --> H[MiniLM embeddings / TF-IDF fallback]
  D --> Q[quality checks]
  G --> I[FastAPI /predict]
  G --> J[FastAPI /predict/timely]
  H --> K[FastAPI /search /chat]
  F --> L[FastAPI /sql /chat]
```

| Concern | Implementation |
|---|---|
| Landing | `data/raw` |
| Bronze / silver / gold | Parquet under `data/` |
| SQL engine | Cached DuckDB over gold |
| Model tracking | MLflow (local SQLite) |
| Serving | uvicorn / Docker |
| Retrieval | MiniLM embeddings (joblib); TF-IDF fallback |

## Gold star schema

**Fact:** `fact_complaints` (grain = one complaint)

**Dims:** `dim_company`, `dim_product`, `dim_issue`, `dim_geo`, `dim_channel`, `dim_date`

DDL: `sql/01_star_schema.sql`  
Analytics: `sql/02_analytical_queries.sql`  
Quality: `etl/quality.py`

## Request routing

```mermaid
flowchart TD
  Q[User question] --> R{looks_analytical?}
  R -->|yes| S[text-to-SQL templates / optional LLM]
  S --> T[SELECT-only guard]
  T --> U[Cached DuckDB on gold]
  R -->|no| V[Embedding / TF-IDF similar complaints]
  V --> W[Cited snippets]
```

## Security notes

- Text-to-SQL rejects multi-statement and DML/DDL verbs before execution.
- Optional `OPENAI_API_KEY` only proposes SQL; the same guard still applies.
- `.env`, raw dumps, and `.joblib` model binaries are gitignored.
