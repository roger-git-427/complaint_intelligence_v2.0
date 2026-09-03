"""English -> read-only DuckDB on gold. Templates offline; optional OPENAI_API_KEY."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from etl.run_queries import connect
from paths import SCHEMA_SQL

ROW_LIMIT = 50
JOIN_SQL = """
FROM fact_complaints AS f
JOIN dim_company AS c ON c.company_key = f.company_key
JOIN dim_product AS p ON p.product_key = f.product_key
JOIN dim_issue AS i ON i.issue_key = f.issue_key
JOIN dim_geo AS g ON g.geo_key = f.geo_key
JOIN dim_date AS d ON d.date_key = f.date_key
"""
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|copy|pragma|export|load|install|call|execute|merge)\b",
    re.I,
)
US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
PRODUCT_ALIASES = {
    "credit cards": "Credit card",
    "credit card": "Credit card",
    "credit reporting": "Credit reporting or other personal consumer reports",
    "credit reports": "Credit reporting or other personal consumer reports",
    "credit report": "Credit reporting or other personal consumer reports",
    "mortgage": "Mortgage",
    "student loans": "Student loan",
    "student loan": "Student loan",
    "debt collection": "Debt collection",
    "checking": "Checking or savings account",
    "savings": "Checking or savings account",
    "bank account": "Checking or savings account",
    "payday": "Payday loan, title loan, personal loan, or advance loan",
    "vehicle": "Vehicle loan or lease",
    "money transfer": "Money transfer, virtual currency, or money service",
}


@dataclass
class SqlResult:
    sql: str
    rows: list[dict]
    source: str
    error: str | None = None


def _extract_state(question: str) -> str | None:
    lowered = question.lower()
    for name, code in sorted(US_STATES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            return code
    upper = question.upper()
    for code in US_STATES.values():
        if re.search(rf"\b{code}\b", upper):
            return code
    return None


def _extract_product(question: str) -> str | None:
    lowered = question.lower()
    for alias, product in sorted(PRODUCT_ALIASES.items(), key=lambda item: -len(item[0])):
        if alias in lowered:
            return product
    return None


def _extract_company(question: str, companies: list[str]) -> str | None:
    lowered = question.lower()
    ranked: list[tuple[int, str]] = []
    for name in companies:
        token = name.lower().split(",")[0].strip()
        if len(token) >= 5 and token in lowered:
            ranked.append((len(token), name))
    return sorted(ranked, reverse=True)[0][1] if ranked else None


def _esc(value: str) -> str:
    return value.replace("'", "''")


def _filters(product: str | None, state: str | None, company: str | None) -> str:
    clauses: list[str] = []
    if product:
        clauses.append(f"p.product = '{_esc(product)}'")
    if state:
        clauses.append(f"g.state = '{state}'")
    if company:
        clauses.append(f"c.company_name = '{_esc(company)}'")
    return ("WHERE " + " AND ".join(clauses)) if clauses else ""


def compile_template(question: str, companies: list[str]) -> tuple[str, str]:
    product = _extract_product(question)
    state = _extract_state(question)
    company = _extract_company(question, companies)
    lowered = question.lower()
    where = _filters(product, state, company)

    if any(word in lowered for word in ("timely", "on time", "response rate")):
        return (
            f"SELECT c.company_name, COUNT(*) AS complaints, "
            f"ROUND(AVG(CAST(f.timely_response AS INTEGER)), 3) AS timely_rate "
            f"{JOIN_SQL} {where} GROUP BY c.company_name ORDER BY complaints DESC LIMIT 15",
            "template:timely_rate",
        )
    if any(word in lowered for word in ("issue", "why", "reason", "problem")):
        return (
            f"SELECT p.product, i.issue, COUNT(*) AS complaints "
            f"{JOIN_SQL} {where} GROUP BY p.product, i.issue ORDER BY complaints DESC LIMIT 15",
            "template:issues",
        )
    if any(word in lowered for word in ("month", "trend", "mom", "over time")):
        return (
            f"SELECT d.year, d.month, p.product, COUNT(*) AS complaints "
            f"{JOIN_SQL} {where} GROUP BY d.year, d.month, p.product "
            f"ORDER BY d.year, d.month, complaints DESC LIMIT 24",
            "template:trend",
        )
    if any(word in lowered for word in ("state", "where", "which state")) and not state:
        return (
            f"SELECT g.state, COUNT(*) AS complaints {JOIN_SQL} {where} "
            f"GROUP BY g.state ORDER BY complaints DESC LIMIT 15",
            "template:states",
        )
    if any(word in lowered for word in ("how many", "count", "volume", "number of")) and not any(
        word in lowered for word in ("which", "top", "most")
    ):
        return f"SELECT COUNT(*) AS complaints {JOIN_SQL} {where}", "template:count"
    return (
        f"SELECT c.company_name, p.product, COUNT(*) AS complaints {JOIN_SQL} {where} "
        f"GROUP BY c.company_name, p.product ORDER BY complaints DESC LIMIT 15",
        "template:top_companies",
    )


def llm_sql(question: str) -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    prompt = (
        "Write one DuckDB SQL query for this CFPB gold star schema. "
        "SELECT/WITH only. Use fact_complaints, dim_company, dim_product, "
        "dim_issue, dim_geo, dim_channel, dim_date. SQL only, no markdown.\n\n"
        f"{SCHEMA_SQL.read_text(encoding='utf-8')[:4000]}\n\nQuestion: {question}"
    )
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You emit only DuckDB SQL."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=45,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip()
    return re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.I).strip()


def guard_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not allowed")
    if FORBIDDEN.search(cleaned):
        raise ValueError("Only read-only SELECT/WITH queries are allowed")
    head = cleaned.lstrip().lower()
    if not (head.startswith("select") or head.startswith("with")):
        raise ValueError("Query must start with SELECT or WITH")
    return cleaned


def run_question(question: str, prefer_llm: bool = True) -> SqlResult:
    con = connect(cached=True)
    sql: str | None = None
    source = "template"
    if prefer_llm:
        try:
            sql = llm_sql(question)
            if sql:
                source = "llm"
        except Exception as exc:
            print(f"LLM SQL skipped: {exc}")
    if not sql:
        companies = (
            con.execute("SELECT company_name FROM dim_company")
            .fetchdf()["company_name"]
            .dropna()
            .astype(str)
            .tolist()
        )
        sql, source = compile_template(question, companies)
    try:
        guarded = guard_sql(sql)
        frame = con.execute(f"SELECT * FROM ({guarded}) AS q LIMIT {ROW_LIMIT}").fetchdf()
        rows = frame.where(pd.notnull(frame), None).to_dict(orient="records")
        return SqlResult(sql=guarded, rows=rows, source=source)
    except Exception as exc:
        return SqlResult(sql=sql or "", rows=[], source=source, error=str(exc))
