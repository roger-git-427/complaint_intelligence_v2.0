from etl.schema import COLUMN_RENAME, SILVER_COLUMNS


def test_complaint_what_happened_maps_to_narrative():
    assert COLUMN_RENAME["complaint_what_happened"] == "narrative"
    assert COLUMN_RENAME["Consumer complaint narrative"] == "narrative"


def test_silver_columns_are_unique_and_complete():
    assert len(SILVER_COLUMNS) == len(set(SILVER_COLUMNS))
    assert "complaint_id" in SILVER_COLUMNS
    assert "narrative" in SILVER_COLUMNS
    assert "timely_response" in SILVER_COLUMNS
