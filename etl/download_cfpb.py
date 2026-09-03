"""Land a CFPB extract into data/raw (API, bulk zip, or Hugging Face snapshot)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RAW_DIR, ROOT

BULK_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
HF_SHARD = (
    "https://huggingface.co/datasets/davidheineman/"
    "consumer-finance-complaints-large/resolve/main/data/train-00000-of-00011.parquet"
)
NARRATIVE_COLS = ("Consumer complaint narrative", "complaint_what_happened", "narrative")
DATE_COLS = ("Date received", "date_received")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/csv,*/*",
    "Referer": "https://www.consumerfinance.gov/data-research/consumer-complaints/",
}


def _scalar(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _col(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"None of {names} in {list(frame.columns)}")


def filter_extract(
    frame: pd.DataFrame, years: list[int], max_records: int, has_narrative: bool
) -> pd.DataFrame:
    received = pd.to_datetime(frame[_col(frame, DATE_COLS)], errors="coerce")
    sliced = frame.loc[received.dt.year.isin(years)]
    if has_narrative:
        text = sliced[_col(sliced, NARRATIVE_COLS)].fillna("").astype(str).str.strip()
        sliced = sliced.loc[text.str.len() > 40]
    if sliced.empty:
        return sliced
    return sliced.head(max_records).reset_index(drop=True)


def download_api(years: list[int], max_records: int, has_narrative: bool) -> pd.DataFrame:
    rows: list[dict] = []
    session = requests.Session()
    fr = 0
    page_size = 1000
    print(f"Fetching CFPB API {min(years)}-{max(years)} (max {max_records})...")
    while len(rows) < max_records:
        params = {
            "size": page_size,
            "frm": fr,
            "date_received_min": f"{min(years)}-01-01",
            "date_received_max": f"{max(years)}-12-31",
            "no_aggs": "true",
            "format": "json",
        }
        if has_narrative:
            params["has_narrative"] = "true"
        response = session.get(API_URL, params=params, headers=HEADERS, timeout=60)
        response.raise_for_status()
        hits = response.json().get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            src = hit.get("_source", hit)
            rows.append({key: _scalar(val) for key, val in src.items()})
            if len(rows) >= max_records:
                break
        fr += page_size
        print(f"  landed {len(rows)} rows")
        if len(hits) < page_size:
            break
    return filter_extract(pd.DataFrame(rows), years, max_records, has_narrative) if rows else pd.DataFrame()


def download_huggingface(years: list[int], max_records: int, has_narrative: bool) -> pd.DataFrame:
    print("Reading Hugging Face CFPB snapshot (~120MB)...")
    frame = pd.read_parquet(HF_SHARD)
    print(f"  shard rows={len(frame):,}")
    return filter_extract(frame, years, max_records, has_narrative)


def download_bulk(years: list[int], max_records: int, has_narrative: bool) -> pd.DataFrame:
    print("Streaming official CFPB bulk dump...")
    kept: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(BULK_URL, chunksize=50_000, dtype=str, low_memory=False, compression="zip"):
        sliced = filter_extract(chunk, years, max_records - total, has_narrative)
        if sliced.empty:
            continue
        kept.append(sliced)
        total += len(sliced)
        print(f"  kept {total} rows")
        if total >= max_records:
            break
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()


def write_raw(frame: pd.DataFrame, source: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = RAW_DIR / f"cfpb_complaints_{source}_{stamp}.csv"
    latest = RAW_DIR / "cfpb_complaints_latest.csv"
    frame.to_csv(csv_path, index=False)
    frame.to_csv(latest, index=False)
    meta = {
        "row_count": int(len(frame)),
        "source": source,
        "landed_at_utc": stamp,
        "csv": str(csv_path.relative_to(ROOT)),
        "columns": list(frame.columns),
    }
    (RAW_DIR / "cfpb_complaints_latest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(frame)} rows -> {latest}")
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("auto", "api", "bulk", "huggingface"), default="auto")
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--max-records", type=int, default=8_000)
    parser.add_argument("--no-narrative-filter", action="store_true")
    args = parser.parse_args()
    has_narrative = not args.no_narrative_filter
    sources = {"api": download_api, "bulk": download_bulk, "huggingface": download_huggingface}
    order = list(sources) if args.source == "auto" else [args.source]
    frame = pd.DataFrame()
    last_error: Exception | None = None
    used = args.source
    for source in order:
        try:
            print(f"Trying source={source}...")
            frame = sources[source](args.years, args.max_records, has_narrative)
            if not frame.empty:
                used = source
                break
            print(f"  {source} returned 0 rows")
        except (requests.RequestException, OSError, ValueError) as exc:
            last_error = exc
            print(f"  {source} failed: {exc}")
    if frame.empty:
        print(f"Download failed: {last_error}", file=sys.stderr)
        return 1
    write_raw(frame, used)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
