"""Route a question to warehouse SQL or complaint retrieval."""

from __future__ import annotations

from collections import Counter

from rag.retriever import ComplaintIndex
from rag.text_to_sql import run_question

SQL_HINTS = (
    "how many", "which", "top", "most", "rank", "rate", "trend", "month",
    "share", "volume", "count", "average", "timely", "by state", "companies", "issuers",
)
RETRIEVAL_HINTS = (
    "similar", "like this", "happened to me", "my credit", "someone stole",
    "account was", "they charged", "complaint about", "narrative",
)


def looks_analytical(question: str) -> bool:
    lowered = question.lower().strip()
    if len(question) > 280:
        return False
    retrieval_hit = any(hint in lowered for hint in RETRIEVAL_HINTS)
    # Prefer retrieval when the user is describing a complaint story.
    if retrieval_hit:
        return False
    sql_hit = any(hint in lowered for hint in SQL_HINTS)
    return sql_hit or lowered.startswith(("show ", "list ", "rank "))


def _summarize_hits(hits: list[dict]) -> str:
    if not hits:
        return "No similar complaints were found."
    product = Counter(hit["product"] for hit in hits).most_common(1)[0][0]
    companies = ", ".join(name for name, _ in Counter(hit["company"] for hit in hits).most_common(3))
    lines = [f"Closest matches are mostly {product}. Frequent companies: {companies}.", ""]
    for i, hit in enumerate(hits, start=1):
        lines.append(
            f"{i}. {hit['complaint_id']} | {hit['company']} | {hit['product']} "
            f"| {hit['state']} (score {hit['score']})"
        )
        lines.append(f"   {hit['snippet']}\n")
    return "\n".join(lines).strip()


def _summarize_sql(result) -> str:
    if result.error:
        return f"Could not answer from the warehouse: {result.error}"
    if not result.rows:
        return "The warehouse query returned no rows for that question."
    keys = list(result.rows[0].keys())
    if keys == ["complaints"] or (len(keys) == 1 and "count" in keys[0].lower()):
        return f"Count from gold fact table: {list(result.rows[0].values())[0]}."
    lines = [f"Warehouse answer ({result.source}, {len(result.rows)} rows):", ""]
    for row in result.rows[:8]:
        lines.append("- " + "; ".join(f"{k}={v}" for k, v in row.items()))
    return "\n".join(lines)


def answer(question: str, index: ComplaintIndex, k: int = 5) -> dict:
    cleaned = (question or "").strip()
    if not cleaned:
        return {"mode": "error", "answer": "Question is empty.", "citations": []}
    if looks_analytical(cleaned):
        result = run_question(cleaned)
        return {
            "mode": "sql",
            "answer": _summarize_sql(result),
            "sql": result.sql,
            "sql_source": result.source,
            "rows": result.rows,
            "citations": [],
            "error": result.error,
        }
    hits = index.search(cleaned, k=k)
    return {"mode": "retrieval", "answer": _summarize_hits(hits), "citations": hits, "sql": None, "rows": []}
