#!/usr/bin/env python
"""The closed loop: retrain a candidate, benchmark it, promote only if it wins.

This is the monthly cadence in miniature:

  1. Pull the latest reviewed labels out of the QA portal.
  2. Rebuild the training set (keeping the *fixed* benchmark so the gate is fair).
  3. Retrain a candidate adapter for the production model.
  4. Benchmark the candidate on the held-out set.
  5. Promote it to ``data/adapters/current`` ONLY if its % correct beats the
     incumbent's. Otherwise keep the incumbent and log why.

A bad retrain never ships — the DuckDB table decides, not a hunch. Prints a clear
PROMOTED / KEPT decision and exits non-zero only on error (not on KEPT).

Run:  uv run python scripts/retrain.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MANAGE = str(ROOT / "packages" / "qa_portal" / "manage.py")

CHILD_ENV = {
    **os.environ,
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONWARNINGS": "ignore",
    "PYTHONUNBUFFERED": "1",
}


def run(cmd: list[str], desc: str) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    if subprocess.run(cmd, cwd=str(ROOT), env=CHILD_ENV).returncode != 0:
        raise SystemExit(f"[FAILED] {desc}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="newer", help="production model to retrain")
    p.add_argument("--max-train", type=int, default=110)
    p.add_argument("--epochs", type=int, default=2,
                   help="epochs for the candidate (2 matches the demo recipe)")
    args = p.parse_args(argv)

    # Import here so path helpers resolve against the repo.
    from freight_schema.paths import adapter_dir, predictions_dir, training_dir
    from eval.report import _pct_correct, passes_gate

    print("=== Closed-loop retrain ===\n")

    # 1-2. Pull reviewed labels + rebuild training (fixed benchmark stays put).
    qa_real = str(training_dir() / "qa_real.jsonl")
    run([PY, MANAGE, "export_training", "--out", qa_real], "export reviewed labels")
    bench = training_dir() / "benchmark.jsonl"
    if not bench.exists():
        raise SystemExit("No benchmark.jsonl — run `make demo` (or eval prep) first.")
    run([PY, "-m", "finetune.prep_dataset", "--real", qa_real,
         "--synthetic", str(training_dir() / "synthetic.jsonl"),
         "--reuse-benchmark", "--max-train", str(args.max_train), "--force"],
        "rebuild training set")

    # 3-4. Retrain the candidate + benchmark it on the fixed holdout.
    run([PY, "-m", "finetune.train", "--model", args.model,
         "--epochs", str(args.epochs), "--max-train", str(args.max_train)],
        "train candidate")
    run([PY, "-m", "eval.predict", "--model", args.model], "benchmark candidate")

    candidate_pred = predictions_dir() / f"{args.model}.jsonl"
    # Keep the promoted snapshot OUT of predictions/*.jsonl — its rows carry the
    # candidate's model_name, so leaving it beside the per-model files would make
    # eval.report's GROUP BY model_name double-count the incumbent.
    promoted_dir = predictions_dir() / "promoted"
    promoted_dir.mkdir(parents=True, exist_ok=True)
    current_pred = promoted_dir / "current.jsonl"
    current_adapter = adapter_dir("current")
    candidate_adapter = adapter_dir(args.model)

    # 5. Gate + promote.
    cand_pct = _pct_correct(candidate_pred)
    incumbent = current_pred if current_pred.exists() else None
    inc_pct = _pct_correct(current_pred) if incumbent else None

    print("\n" + "=" * 60)
    if passes_gate(candidate_pred, incumbent):
        shutil.rmtree(current_adapter, ignore_errors=True)
        shutil.copytree(candidate_adapter, current_adapter)
        shutil.copyfile(candidate_pred, current_pred)
        if inc_pct is None:
            print(f"  PROMOTED — established baseline at {cand_pct:.1f}% correct.")
        else:
            print(f"  PROMOTED — candidate {cand_pct:.1f}% > incumbent {inc_pct:.1f}%.")
        print(f"  -> {current_adapter}")
    else:
        print(f"  KEPT incumbent — candidate {cand_pct:.1f}% did not beat "
              f"{inc_pct:.1f}%. A bad retrain never ships.")
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
