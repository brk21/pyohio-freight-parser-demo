"""Command-line interface for the synthetic freight-data generator.

Three commands drive the synthetic-data stage of the pipeline::

    freight-synthetic generate --n 500 --seed 0   # build + store in DuckDB
    freight-synthetic query                        # peek at a few stored rows
    freight-synthetic export --out training/synthetic.jsonl

Also runnable as ``python -m synthetic.cli ...``. The store is the DuckDB file at
``freight_schema.paths.synthetic_db()``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from freight_schema import paths

from synthetic import store
from synthetic.generator import generate_examples

app = typer.Typer(
    add_completion=False,
    help="Generate, inspect, and export synthetic freight confirmations.",
)


@app.command()
def generate(
    n: int = typer.Option(500, help="Number of examples to generate."),
    seed: int = typer.Option(0, help="Seed for deterministic generation."),
) -> None:
    """Generate ``n`` examples from ``seed`` and (re)store them in DuckDB."""
    examples = generate_examples(n, seed)
    con = store.connect()
    try:
        store.reset_table(con)
        inserted = store.insert_examples(con, examples)
    finally:
        con.close()
    typer.echo(f"Generated {inserted} examples (seed={seed}) -> {paths.synthetic_db()}")


@app.command()
def query(
    n: int = typer.Option(5, help="How many rows to show."),
) -> None:
    """Show a small sample of stored examples."""
    con = store.connect()
    try:
        total = store.count(con)
        rows = store.sample_examples(con, n)
    finally:
        con.close()

    if total == 0:
        typer.echo("Store is empty. Run 'generate' first.")
        raise typer.Exit(code=0)

    typer.echo(f"{total} examples in {paths.synthetic_db()}\n")
    for row in rows:
        typer.echo(f"[{row['id']}] style={row['style']} complexity={row['complexity']}")
        typer.echo(f"  text : {row['text']!r}")
        typer.echo(f"  gold : {row['parsed_json']}")
        typer.echo("")


@app.command()
def export(
    out: str = typer.Option(
        None,
        help="Output JSONL path (default: data/training/synthetic.jsonl). "
        "Relative paths resolve against the repo root.",
    ),
) -> None:
    """Export all stored examples to JSONL for the training-prep stage."""
    out_path = (
        Path(paths.training_dir() / "synthetic.jsonl") if out is None else Path(out)
    )
    if not out_path.is_absolute():
        out_path = paths.repo_root() / out_path

    con = store.connect()
    try:
        written = store.export_jsonl(con, out_path)
    finally:
        con.close()
    typer.echo(f"Exported {written} examples -> {out_path}")


if __name__ == "__main__":  # enables `python -m synthetic.cli`
    app()
