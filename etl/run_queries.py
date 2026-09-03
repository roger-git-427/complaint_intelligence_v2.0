"""DuckDB connection over gold parquet, with a process-level cache for the API."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import GOLD_DIR, GOLD_TABLES, SQL_QUERIES

_CACHED: duckdb.DuckDBPyConnection | None = None


def _open() -> duckdb.DuckDBPyConnection:
    missing = [name for name in GOLD_TABLES if not (GOLD_DIR / f"{name}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing gold tables {missing}. Run: python etl/transform.py")
    con = duckdb.connect(database=":memory:")
    for name in GOLD_TABLES:
        path = (GOLD_DIR / f"{name}.parquet").as_posix()
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    return con


def connect(*, cached: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection. Use cached=True from the API hot path."""
    global _CACHED
    if not cached:
        return _open()
    if _CACHED is None:
        _CACHED = _open()
    return _CACHED


def reset_cache() -> None:
    global _CACHED
    if _CACHED is not None:
        try:
            _CACHED.close()
        except Exception:
            pass
    _CACHED = None


def split_queries(sql_text: str) -> list[tuple[str, str]]:
    named: list[tuple[str, str]] = []
    for chunk in re.split(r"^-- QUERY:\s*", sql_text, flags=re.MULTILINE):
        lines = [line for line in chunk.strip().splitlines() if line.strip()]
        if not lines or lines[0].startswith("--"):
            continue
        name = lines[0].strip()
        sql = "\n".join(line for line in lines[1:] if not line.startswith("--")).strip().rstrip(";")
        if sql:
            named.append((name, sql))
    return named


def main() -> int:
    reset_cache()
    queries = split_queries(SQL_QUERIES.read_text(encoding="utf-8"))
    con = connect()
    print(f"Gold fact_complaints: {con.execute('SELECT COUNT(*) FROM fact_complaints').fetchone()[0]} rows\n")
    for name, sql in queries:
        print("=" * 72)
        print(name)
        print("=" * 72)
        try:
            frame = con.execute(sql).fetchdf()
        except Exception as exc:
            print(f"FAILED: {exc}\n{sql}\n")
            return 1
        shown = frame.head(20)
        print(shown.to_string(index=False) if not shown.empty else "(no rows)")
        if len(frame) > 20:
            print(f"... {len(frame)} rows total")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
