"""Tests for the DuckDB synthetic store.

These use a temp database file so the real ``data/synthetic.duckdb`` is never
touched.
"""

from __future__ import annotations

import json

from synthetic import store
from synthetic.generator import generate_examples


def test_roundtrip_insert_count_sample_export(tmp_path) -> None:
    examples = generate_examples(25, seed=0)
    db_path = tmp_path / "synthetic.duckdb"

    con = store.connect(db_path)
    try:
        store.reset_table(con)
        inserted = store.insert_examples(con, examples)
        assert inserted == 25
        assert store.count(con) == 25

        sample = store.sample_examples(con, 5)
        assert len(sample) == 5
        assert set(sample[0]) == {"id", "text", "parsed_json", "style", "complexity"}

        out = tmp_path / "export.jsonl"
        written = store.export_jsonl(con, out)
        assert written == 25
    finally:
        con.close()

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 25
    # Export contract: exactly these four fields, parsed_json stays a JSON string.
    for rec in records:
        assert set(rec) == {"id", "text", "parsed_json", "style"}
        assert isinstance(rec["parsed_json"], str)
        assert isinstance(json.loads(rec["parsed_json"]), list)


def test_reset_table_is_idempotent_and_clears(tmp_path) -> None:
    db_path = tmp_path / "synthetic.duckdb"
    con = store.connect(db_path)
    try:
        store.reset_table(con)
        store.insert_examples(con, generate_examples(10, seed=0))
        assert store.count(con) == 10
        # A second reset drops the old rows.
        store.reset_table(con)
        assert store.count(con) == 0
    finally:
        con.close()
