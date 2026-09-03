import pytest

from rag.text_to_sql import compile_template, guard_sql


def test_guard_sql_allows_select():
    assert guard_sql("SELECT 1").startswith("SELECT")


def test_guard_sql_rejects_writes():
    with pytest.raises(ValueError, match="read-only"):
        guard_sql("DROP TABLE fact_complaints")
    with pytest.raises(ValueError, match="Multiple"):
        guard_sql("SELECT 1; SELECT 2")


def test_compile_template_filters_state_and_product():
    sql, source = compile_template(
        "Which companies have the most credit card complaints in Texas?",
        companies=["Example Bank NA"],
    )
    assert source == "template:top_companies"
    assert "Credit card" in sql
    assert "g.state = 'TX'" in sql
    assert "GROUP BY" in sql.upper()


def test_compile_template_count():
    sql, source = compile_template("How many mortgage complaints are there?", companies=[])
    assert source == "template:count"
    assert "COUNT(*)" in sql.upper()
    assert "Mortgage" in sql
