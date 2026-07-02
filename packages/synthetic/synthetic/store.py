"""DuckDB store for synthetic examples.

In production this role — a durable, queryable pool of generated training data —
was a cloud data warehouse; DuckDB is the zero-setup laptop stand-in so the whole
demo runs offline from a single file on disk.

The table mirrors the shared data contract exactly::

    examples(id VARCHAR, text VARCHAR, parsed_json VARCHAR,
             style VARCHAR, complexity INTEGER)

``parsed_json`` holds the canonical JSON *string* of the gold shipment lines (as
produced by :func:`freight_schema.dump_lines`), so what goes into the warehouse
is exactly what the training-prep stage reads back out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import duckdb

from freight_schema import paths

_COLUMNS = ("id", "text", "parsed_json", "style", "complexity")


def connect(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the DuckDB database file.

    Defaults to :func:`freight_schema.paths.synthetic_db`. Callers own the
    returned connection and should ``.close()`` it (or use it in a ``with``).
    """
    path = Path(db_path) if db_path is not None else paths.synthetic_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def reset_table(con: duckdb.DuckDBPyConnection) -> None:
    """(Re)create the ``examples`` table, dropping any existing rows.

    Generation is a full rebuild — the synthetic set is defined entirely by its
    seed — so we drop and recreate rather than append, keeping the store a clean
    snapshot of one generation run.
    """
    con.execute("DROP TABLE IF EXISTS examples")
    con.execute(
        """
        CREATE TABLE examples (
            id VARCHAR,
            text VARCHAR,
            parsed_json VARCHAR,
            style VARCHAR,
            complexity INTEGER
        )
        """
    )


def insert_examples(con: duckdb.DuckDBPyConnection, examples: Iterable[dict]) -> int:
    """Insert example dicts (as produced by the generator). Returns the count."""
    rows = [tuple(ex[col] for col in _COLUMNS) for ex in examples]
    if rows:
        con.executemany("INSERT INTO examples VALUES (?, ?, ?, ?, ?)", rows)
    return len(rows)


def count(con: duckdb.DuckDBPyConnection) -> int:
    """Number of stored examples."""
    return con.execute("SELECT COUNT(*) FROM examples").fetchone()[0]


def sample_examples(con: duckdb.DuckDBPyConnection, n: int = 5) -> list[dict]:
    """Return up to ``n`` rows (ordered by id, so the sample is reproducible)."""
    cursor = con.execute(
        "SELECT id, text, parsed_json, style, complexity "
        "FROM examples ORDER BY id LIMIT ?",
        [n],
    )
    return [dict(zip(_COLUMNS, row)) for row in cursor.fetchall()]


def export_jsonl(con: duckdb.DuckDBPyConnection, out_path: Path | str) -> int:
    """Write all rows to ``out_path`` as JSONL and return the number written.

    Each line matches the shared export contract:
    ``{"id", "text", "parsed_json", "style"}`` — ``parsed_json`` stays a JSON
    *string* (the downstream training-prep stage parses it).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = con.execute(
        "SELECT id, text, parsed_json, style FROM examples ORDER BY id"
    ).fetchall()
    with out.open("w", encoding="utf-8") as fh:
        for id_, text, parsed_json, style in rows:
            record = {"id": id_, "text": text, "parsed_json": parsed_json, "style": style}
            fh.write(json.dumps(record) + "\n")
    return len(rows)
