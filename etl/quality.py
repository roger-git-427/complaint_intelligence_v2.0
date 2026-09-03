"""Lightweight data-quality checks on silver + gold parquet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import GOLD_DIR, QUALITY_REPORT_PATH, SILVER_PATH


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def run_checks() -> dict:
    checks: list[dict] = []
    if not SILVER_PATH.exists():
        raise FileNotFoundError(f"Missing {SILVER_PATH}. Run: python etl/transform.py")

    silver = pd.read_parquet(SILVER_PATH)
    checks.append(_check("silver_not_empty", len(silver) > 0, f"rows={len(silver)}"))
    checks.append(
        _check(
            "silver_unique_complaint_id",
            silver["complaint_id"].nunique() == len(silver),
            f"unique={silver['complaint_id'].nunique()} total={len(silver)}",
        )
    )
    null_required = silver[["complaint_id", "date_received", "product", "company"]].isna().any().any()
    checks.append(_check("silver_required_non_null", not null_required, "complaint_id/date/product/company"))
    checks.append(
        _check(
            "silver_has_narrative_rate",
            float(silver["has_narrative"].mean()) > 0.5,
            f"rate={float(silver['has_narrative'].mean()):.3f}",
        )
    )

    fact_path = GOLD_DIR / "fact_complaints.parquet"
    company_path = GOLD_DIR / "dim_company.parquet"
    if not fact_path.exists() or not company_path.exists():
        raise FileNotFoundError(f"Missing gold tables under {GOLD_DIR}. Run: python etl/transform.py")

    fact = pd.read_parquet(fact_path)
    company = pd.read_parquet(company_path)
    checks.append(_check("gold_fact_not_empty", len(fact) > 0, f"rows={len(fact)}"))
    checks.append(
        _check(
            "gold_fact_matches_silver",
            len(fact) == len(silver),
            f"fact={len(fact)} silver={len(silver)}",
        )
    )
    orphan_companies = ~fact["company_key"].isin(set(company["company_key"]))
    checks.append(
        _check(
            "gold_company_fk",
            not orphan_companies.any(),
            f"orphans={int(orphan_companies.sum())}",
        )
    )
    checks.append(
        _check(
            "gold_days_to_company_finite",
            bool(fact["days_to_company"].dropna().between(-1, 365).mean() > 0.95)
            if fact["days_to_company"].notna().any()
            else True,
            "mostly within [-1, 365] days",
        )
    )

    report = {
        "passed": all(item["ok"] for item in checks),
        "checks": checks,
        "silver_rows": int(len(silver)),
        "gold_fact_rows": int(len(fact)),
    }
    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    report = run_checks()
    for item in report["checks"]:
        mark = "PASS" if item["ok"] else "FAIL"
        print(f"[{mark}] {item['check']}: {item['detail']}")
    print(f"Wrote {QUALITY_REPORT_PATH}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
