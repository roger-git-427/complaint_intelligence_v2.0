from etl import transform
from etl.transform import build_dims, build_fact, to_silver
from tests.conftest import sample_raw_frame


def test_to_silver_dedupes_and_types(tmp_path, monkeypatch):
    monkeypatch.setattr(transform, "SILVER_DIR", tmp_path / "silver")
    silver = to_silver(sample_raw_frame())
    assert len(silver) == 2
    assert set(silver["complaint_id"]) == {"1001", "1002"}
    assert silver.loc[silver["complaint_id"] == "1001", "date_received"].iloc[0].day == 3
    assert bool(silver.loc[silver["complaint_id"] == "1002", "timely_response"].iloc[0]) is False
    assert silver["has_narrative"].all()
    assert (tmp_path / "silver" / "part-0000.parquet").exists()


def test_gold_star_schema_keys_line_up(tmp_path, monkeypatch):
    monkeypatch.setattr(transform, "SILVER_DIR", tmp_path / "silver")
    silver = to_silver(sample_raw_frame())
    dims = build_dims(silver)
    fact = build_fact(silver, dims)
    assert len(fact) == 2
    assert set(dims) == {
        "dim_company",
        "dim_product",
        "dim_issue",
        "dim_geo",
        "dim_channel",
        "dim_date",
    }
    assert fact["company_key"].isin(dims["dim_company"]["company_key"]).all()
    assert fact["product_key"].isin(dims["dim_product"]["product_key"]).all()
    assert fact["date_key"].isin(dims["dim_date"]["date_key"]).all()
