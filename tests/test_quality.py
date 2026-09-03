from etl import quality, transform
from etl.quality import run_checks
from etl.transform import build_dims, build_fact, to_silver, write_gold
from tests.conftest import sample_raw_frame


def test_quality_checks_pass_on_tiny_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(transform, "SILVER_DIR", tmp_path / "silver")
    monkeypatch.setattr(transform, "GOLD_DIR", tmp_path / "gold")
    monkeypatch.setattr(quality, "SILVER_PATH", tmp_path / "silver" / "part-0000.parquet")
    monkeypatch.setattr(quality, "GOLD_DIR", tmp_path / "gold")
    monkeypatch.setattr(quality, "QUALITY_REPORT_PATH", tmp_path / "quality_report.json")

    silver = to_silver(sample_raw_frame())
    dims = build_dims(silver)
    write_gold(dims, build_fact(silver, dims))

    report = run_checks()
    assert report["passed"] is True
    assert (tmp_path / "quality_report.json").exists()
