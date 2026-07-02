"""The release gate: one DuckDB query decides what ships.

Reads the scored prediction JSONL files straight off disk with DuckDB and emits
(a) a per-category breakdown for each model — so you can see *where* it fails —
and (b) a cross-model comparison table ranked by % correct, exactly the query
that gates a release in production.

Run:  uv run python -m eval.report
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import sys
from pathlib import Path

from freight_schema.paths import predictions_dir

from eval.score import CATEGORIES

# Only the scalar columns are read from the JSONL; the nested expected/predicted
# arrays are ignored here (they're for debugging, in the raw/ files).
_COLUMNS = ("{'model_name':'VARCHAR','version':'VARCHAR','category':'VARCHAR',"
            "'duration':'DOUBLE','field_accuracy':'DOUBLE'}")

# The headline release-gate query (printed in the report for teaching value).
# Ranked by exact % correct (the gate), then by field-level accuracy — the finer
# metric that reveals the generational gap when exact-match ties at the low end.
GATE_SQL = """\
SELECT model_name, version,
       100 * avg((category = 'CORRECT')::INT)::FLOAT AS pct_correct,
       100 * avg(field_accuracy)::FLOAT           AS pct_fields,
       count(*) AS n,
       avg(duration)::DECIMAL(10,2)               AS avg_seconds
FROM read_json('{glob}', columns={cols}, format='newline_delimited')
GROUP BY model_name, version
ORDER BY pct_correct DESC, pct_fields DESC;"""


def _duckdb():
    import duckdb
    return duckdb


def comparison_table(glob: str) -> list[tuple]:
    """Run the cross-model gate query; return rows (already ranked)."""
    sql = GATE_SQL.format(glob=glob, cols=_COLUMNS)
    return _duckdb().sql(sql).fetchall()


def category_breakdown(path: Path) -> list[tuple[str, int]]:
    """Category -> count for a single model's scored file (precedence order)."""
    with path.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    counts = {c: 0 for c in CATEGORIES}
    for r in rows:
        counts[r.get("category", "OTHER")] = counts.get(r.get("category", "OTHER"), 0) + 1
    return [(c, counts[c]) for c in CATEGORIES if counts[c]]


def _pct_correct(path: Path) -> float:
    with path.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        return 0.0
    return 100.0 * sum(1 for r in rows if r.get("category") == "CORRECT") / len(rows)


def passes_gate(candidate: Path, incumbent: Path | None) -> bool:
    """True only if the candidate's % correct strictly beats the incumbent's.

    A missing incumbent means there is nothing to beat yet -> the candidate ships.
    """
    if incumbent is None or not Path(incumbent).exists():
        return True
    return _pct_correct(Path(candidate)) > _pct_correct(Path(incumbent))


def _print_table(headers: list[str], rows: list[tuple]) -> None:
    cols = [headers] + [[_fmt(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cols) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(_fmt(c).ljust(widths[i]) for i, c in enumerate(row)))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def print_report(glob: str) -> None:
    files = [Path(p) for p in sorted(globlib.glob(glob))]
    if not files:
        print(f"No prediction files match {glob!r}. Run `eval.predict` first.")
        return

    print("\n=== Per-category breakdown ===")
    for f in files:
        model = f.stem
        print(f"\n[{model}]")
        breakdown = category_breakdown(f)
        _print_table(["category", "count"], breakdown)

    print("\n=== Release gate: cross-model comparison (DuckDB) ===")
    print(GATE_SQL.format(glob=glob, cols=_COLUMNS))
    print()
    rows = comparison_table(glob)
    _print_table(
        ["model_name", "version", "pct_correct", "pct_fields", "n", "avg_seconds"],
        rows,
    )
    if rows:
        winner = rows[0]
        print(f"\n-> Ranked #1: {winner[0]} — {winner[2]:.1f}% exact, "
              f"{winner[3]:.1f}% fields correct. The table decides what ships, not a hunch.")
        print("   (Exact-match is a harsh bar for a 0.5B on CPU; field accuracy shows "
              "the newer base pulling ahead. Numbers illustrative — production runs an 8B.)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions",
        default=str(predictions_dir() / "*.jsonl"),
        help="Glob of scored prediction JSONL files.",
    )
    args = p.parse_args(argv)
    print_report(args.predictions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
