"""Bronze (passthrough) -> silver (typed, deduped) -> gold star schema parquet."""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from etl.schema import COLUMN_RENAME, NULL_TOKENS, SILVER_COLUMNS
from paths import BRONZE_DIR, GOLD_DIR, RAW_DIR, ROOT, SILVER_DIR

JOIN_KEYS = ("company", "product", "sub_product", "issue", "sub_issue", "state", "zip_code", "submitted_via")


def _surrogate_key(*parts: str) -> str:
    material = "|".join((part or "").strip().lower() for part in parts)
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def _nullify(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.str.lower().isin(NULL_TOKENS))


def _yes_no_to_bool(series: pd.Series) -> pd.Series:
    mapped = series.astype("string").str.strip().str.lower().map(
        {"yes": True, "y": True, "true": True, "no": False, "n": False, "false": False}
    )
    return mapped.astype("boolean")


def _replace_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _join_keys(silver: pd.DataFrame) -> pd.DataFrame:
    keyed = silver.copy()
    for column in JOIN_KEYS:
        keyed[column] = keyed[column].fillna("")
    return keyed


def _dim(silver: pd.DataFrame, cols: list[str], prefix: str, key_name: str) -> pd.DataFrame:
    out = silver[cols].drop_duplicates().copy()
    out[key_name] = [
        _surrogate_key(prefix, *[str(v or "") for v in row])
        for row in out.itertuples(index=False, name=None)
    ]
    return out.sort_values(cols).reset_index(drop=True)


def load_raw() -> pd.DataFrame:
    latest = RAW_DIR / "cfpb_complaints_latest.csv"
    if not latest.exists():
        raise FileNotFoundError(f"Missing {latest}. Run: python etl/download_cfpb.py")
    return pd.read_csv(latest, dtype=str, low_memory=False)


def to_bronze(raw: pd.DataFrame) -> pd.DataFrame:
    _replace_dir(BRONZE_DIR)
    raw.to_parquet(BRONZE_DIR / "part-0000.parquet", index=False)
    return raw


def to_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    silver = bronze.rename(columns=COLUMN_RENAME)
    for column in SILVER_COLUMNS:
        if column not in silver.columns:
            silver[column] = pd.NA
    silver = silver[SILVER_COLUMNS].copy()
    for column in silver.columns:
        silver[column] = _nullify(silver[column])

    before = len(silver)
    silver = silver.dropna(subset=["complaint_id"]).drop_duplicates(subset=["complaint_id"], keep="last")
    silver["date_received"] = pd.to_datetime(silver["date_received"], errors="coerce", utc=True)
    silver["date_sent_to_company"] = pd.to_datetime(silver["date_sent_to_company"], errors="coerce", utc=True)
    invalid_dates = int(silver["date_received"].isna().sum())
    silver = silver.dropna(subset=["date_received", "product", "company"])
    silver["timely_response"] = _yes_no_to_bool(silver["timely_response"])
    silver["consumer_disputed"] = _yes_no_to_bool(silver["consumer_disputed"])
    silver["days_to_company"] = (
        silver["date_sent_to_company"] - silver["date_received"]
    ).dt.total_seconds() / 86400.0
    silver["has_narrative"] = silver["narrative"].fillna("").str.len().gt(40)
    silver["ingested_at"] = pd.Timestamp.now(tz="UTC")

    _replace_dir(SILVER_DIR)
    silver.to_parquet(SILVER_DIR / "part-0000.parquet", index=False)
    print(f"Silver: {len(silver)} rows (dropped {before - len(silver)}, {invalid_dates} bad dates)")
    return silver


def build_dims(silver: pd.DataFrame) -> dict[str, pd.DataFrame]:
    silver = _join_keys(silver)
    company = _dim(silver, ["company"], "company", "company_key").rename(columns={"company": "company_name"})
    product = _dim(silver, ["product", "sub_product"], "product", "product_key")
    issue = _dim(silver, ["issue", "sub_issue"], "issue", "issue_key")
    geo = _dim(silver, ["state", "zip_code"], "geo", "geo_key")
    channel = silver[["submitted_via"]].replace("", "unknown").fillna("unknown")
    channel = _dim(channel.rename(columns={"submitted_via": "channel_name"}), ["channel_name"], "channel", "channel_key")

    dates = pd.date_range(
        silver["date_received"].min().normalize(),
        silver["date_received"].max().normalize(),
        freq="D",
        tz="UTC",
    )
    dim_date = pd.DataFrame({"date": dates})
    dim_date["date_key"] = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["year"] = dim_date["date"].dt.year
    dim_date["quarter"] = dim_date["date"].dt.quarter
    dim_date["month"] = dim_date["date"].dt.month
    dim_date["month_name"] = dim_date["date"].dt.strftime("%B")
    dim_date["week"] = dim_date["date"].dt.isocalendar().week.astype(int)
    dim_date["day_of_week"] = dim_date["date"].dt.day_name()
    dim_date["is_weekend"] = dim_date["date"].dt.dayofweek.ge(5)
    return {
        "dim_company": company,
        "dim_product": product,
        "dim_issue": issue,
        "dim_geo": geo,
        "dim_channel": channel,
        "dim_date": dim_date,
    }


def build_fact(silver: pd.DataFrame, dims: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = _join_keys(silver)
    frame = frame.merge(dims["dim_company"], left_on="company", right_on="company_name", how="left")
    frame = frame.merge(dims["dim_product"], on=["product", "sub_product"], how="left")
    frame = frame.merge(dims["dim_issue"], on=["issue", "sub_issue"], how="left")
    frame = frame.merge(dims["dim_geo"], on=["state", "zip_code"], how="left")
    frame["channel_name"] = frame["submitted_via"].replace("", "unknown")
    frame = frame.merge(dims["dim_channel"], on="channel_name", how="left")
    frame["date_key"] = frame["date_received"].dt.strftime("%Y%m%d").astype(int)
    frame["sent_date_key"] = frame["date_sent_to_company"].dt.strftime("%Y%m%d").astype("Int64")
    cols = [
        "complaint_id", "company_key", "product_key", "issue_key", "geo_key",
        "channel_key", "date_key", "sent_date_key", "date_received",
        "date_sent_to_company", "days_to_company", "timely_response",
        "consumer_disputed", "has_narrative", "company_response", "consumer_consent",
    ]
    return frame[cols].drop_duplicates(subset=["complaint_id"], keep="last").reset_index(drop=True)


def write_gold(dims: dict[str, pd.DataFrame], fact: pd.DataFrame) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in {**dims, "fact_complaints": fact}.items():
        path = GOLD_DIR / f"{name}.parquet"
        table.to_parquet(path, index=False)
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"Gold {name}: {len(table)} rows -> {shown}")


def main() -> int:
    raw = load_raw()
    print(f"Raw: {len(raw)} rows, {len(raw.columns)} columns")
    bronze = to_bronze(raw)
    print(f"Bronze: {len(bronze)} rows")
    silver = to_silver(bronze)
    dims = build_dims(silver)
    write_gold(dims, build_fact(silver, dims))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
