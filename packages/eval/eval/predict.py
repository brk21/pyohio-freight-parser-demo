"""Run a model over the benchmark set and write scored predictions.

Uses the *same* constrained-decoding inference path the API serves
(``serving.inference.parse``), so the benchmark measures exactly what ships. For
each held-out confirmation we record the expected gold, the model's prediction,
and the wall-clock decode time, then score each row into a category.

Run:  uv run python -m eval.predict --model newer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from freight_schema import dump_lines
from freight_schema.paths import predictions_dir, training_dir

from eval.score import score_predictions

DEFAULT_VERSION = os.environ.get("FREIGHT_VERSION", "v0.1.0")


def _read_benchmark(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(model: str, benchmark: Path, version: str = DEFAULT_VERSION) -> Path:
    """Predict + score the benchmark for one model. Returns the scored file path."""
    from serving.inference import parse, warmup

    rows = _read_benchmark(benchmark)
    print(f"[predict:{model}] {len(rows)} benchmark rows (version {version})")

    # Warm the model + compile its decoder once, so recorded durations reflect
    # steady-state inference rather than one-time FSM compilation.
    warmup(model)

    raw: list[dict] = []
    for row in rows:
        record = {
            "id": row.get("id"),
            "model_name": model,
            "version": version,
            "text": row["text"],
            "expected": row["expected"],
            "predicted": None,
            "duration": None,
            "error": None,
        }
        try:
            result = parse(row["text"], model=model)
            record["predicted"] = json.loads(dump_lines(result.confirmation.root))
            record["duration"] = round(result.duration, 3)
            record["effective_model"] = result.model
            record["fell_back"] = result.fell_back
        except Exception as exc:  # a real production runner logs and moves on
            record["error"] = f"{type(exc).__name__}: {exc}"
        raw.append(record)

    # Persist the raw run (audit trail) and the scored run (what the gate reads).
    raw_dir = predictions_dir() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{model}.jsonl"
    with raw_path.open("w") as fh:
        for r in raw:
            fh.write(json.dumps(r) + "\n")

    scored = score_predictions(raw)
    scored_path = predictions_dir() / f"{model}.jsonl"
    with scored_path.open("w") as fh:
        for r in scored:
            fh.write(json.dumps(r) + "\n")

    n_correct = sum(1 for r in scored if r["category"] == "CORRECT")
    n_err = sum(1 for r in scored if r["category"] == "ERROR")
    print(f"[predict:{model}] {n_correct}/{len(scored)} CORRECT, {n_err} ERROR "
          f"-> {scored_path}")
    return scored_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="newer")
    p.add_argument("--benchmark", type=Path, default=training_dir() / "benchmark.jsonl")
    p.add_argument("--version", default=DEFAULT_VERSION)
    args = p.parse_args(argv)
    run(model=args.model, benchmark=args.benchmark, version=args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
