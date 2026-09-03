from etl import run_queries


def test_duckdb_cache_reuses_connection(monkeypatch, tmp_path):
    calls = {"n": 0}
    real_open = run_queries._open

    def counting_open():
        calls["n"] += 1
        # Avoid needing gold parquet: return a tiny in-memory duckdb
        import duckdb

        return duckdb.connect(database=":memory:")

    monkeypatch.setattr(run_queries, "_open", counting_open)
    run_queries.reset_cache()
    first = run_queries.connect(cached=True)
    second = run_queries.connect(cached=True)
    assert first is second
    assert calls["n"] == 1
    run_queries.reset_cache()
    _ = real_open  # keep import used for type checkers / clarity
