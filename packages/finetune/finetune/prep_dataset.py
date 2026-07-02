"""Turn reviewed real + synthetic examples into an alpaca training set.

Merges the QA-portal export (trusted, human-reviewed labels) with a configurable
ratio of synthetic examples, holds out a random slice as a benchmark set, and
writes alpaca-format JSONL. The prompt used for training is composed with the
shared ``build_prompt`` so it is byte-for-byte identical to what serving sends.

Why a ratio? Synthetic data fills the long tail of rare formats, but too much of
it drowns out the real signal. The talk lands on roughly 4:1 synthetic:real —
enough coverage without dilution. It's an empirical knob, exposed here as
``--synthetic-ratio``.

Run:  uv run python -m finetune.prep_dataset [--synthetic-ratio 4] [--holdout-percent 15]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from freight_schema import INSTRUCTION
from freight_schema.models import build_prompt  # noqa: F401  (kept explicit for clarity)
from freight_schema.paths import training_dir


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def prepare(
    real_path: Path,
    synthetic_path: Path,
    out_dir: Path,
    synthetic_ratio: float = 4.0,
    holdout_percent: float = 15.0,
    holdout_count: int | None = None,
    max_train: int | None = None,
    seed: int = 0,
    force: bool = False,
    reuse_benchmark: bool = False,
) -> dict:
    """Build ``train.jsonl`` (alpaca) + ``benchmark.jsonl`` (held-out gold).

    With ``reuse_benchmark`` the existing benchmark is left untouched and *all*
    rows go to training — used by the retrain loop so the candidate and incumbent
    are always judged on the same fixed holdout. Returns a small summary dict.
    """
    train_out = out_dir / "train.jsonl"
    bench_out = out_dir / "benchmark.jsonl"
    to_write = [train_out] if reuse_benchmark else [train_out, bench_out]
    for existing in to_write:
        if existing.exists() and not force:
            raise FileExistsError(
                f"{existing} already exists; pass --force to overwrite."
            )

    rng = random.Random(seed)
    real = _read_jsonl(real_path)
    synthetic = _read_jsonl(synthetic_path)

    # Choose how many synthetic rows to mix in: ratio * (number of real rows).
    # If there aren't enough synthetic rows, use them all (and note it).
    if real:
        target_syn = round(len(real) * synthetic_ratio)
        if len(synthetic) > target_syn:
            synthetic = rng.sample(synthetic, target_syn)
    # (If we have no real rows yet — e.g. before any QA — we simply train on
    #  synthetic alone so the pipeline still runs end to end.)

    combined = [{**r, "source": "real"} for r in real] + [
        {**s, "source": "synthetic"} for s in synthetic
    ]
    rng.shuffle(combined)

    # Hold out a benchmark set. A fixed count (if given) keeps `make demo` fast
    # and predictable; otherwise fall back to a percentage.
    if reuse_benchmark:
        # Retrain loop: keep the existing benchmark and send everything else to
        # training — but EXCLUDE the held-out rows from the training pool. The
        # benchmark was carved from this same (deterministic) shuffle, so without
        # this filter the candidate would train on its own gate set (train/test
        # contamination) and a worse candidate could sail through passes_gate.
        held_out_text = set()
        if bench_out.exists():
            with bench_out.open() as fh:
                held_out_text = {json.loads(line)["text"] for line in fh if line.strip()}
        combined = [row for row in combined if row["text"] not in held_out_text]
        n_holdout = 0
    else:
        n_holdout = holdout_count if holdout_count is not None else max(
            1, round(len(combined) * holdout_percent / 100.0)
        )
    n_holdout = min(n_holdout, len(combined))
    benchmark_rows = combined[:n_holdout]
    train_rows = combined[n_holdout:]
    if max_train is not None:
        train_rows = train_rows[:max_train]

    # Alpaca format. `instruction` is the shared INSTRUCTION (so the Axolotl
    # alpaca handler and our trl trainer agree); the trl trainer composes the
    # actual prompt with build_prompt(input) for exact train/serve parity.
    alpaca = [
        {"instruction": INSTRUCTION, "input": row["text"], "output": row["parsed_json"]}
        for row in train_rows
    ]
    benchmark = [
        {"id": row.get("id", f"bench-{i}"), "text": row["text"],
         "expected": json.loads(row["parsed_json"])}
        for i, row in enumerate(benchmark_rows)
    ]

    _write_jsonl(train_out, alpaca)
    if not reuse_benchmark:
        _write_jsonl(bench_out, benchmark)

    return {
        "real": len(real),
        "synthetic_used": len(synthetic),
        "train": len(alpaca),
        "benchmark": len(benchmark),
        "train_path": str(train_out),
        "benchmark_path": str(bench_out),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    td = training_dir()
    p.add_argument("--real", type=Path, default=td / "qa_real.jsonl",
                   help="QA-portal export JSONL (real, reviewed labels).")
    p.add_argument("--synthetic", type=Path, default=td / "synthetic.jsonl",
                   help="Synthetic export JSONL.")
    p.add_argument("--out-dir", type=Path, default=td)
    p.add_argument("--synthetic-ratio", type=float, default=4.0,
                   help="Synthetic:real mixing ratio (default 4:1).")
    p.add_argument("--holdout-percent", type=float, default=15.0)
    p.add_argument("--holdout-count", type=int, default=None,
                   help="Exact benchmark size (overrides --holdout-percent).")
    p.add_argument("--max-train", type=int, default=None,
                   help="Cap training rows (keeps the CPU demo fast).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--reuse-benchmark", action="store_true",
                   help="Keep the existing benchmark; send all rows to training "
                        "(used by the retrain loop for a fair, fixed gate).")
    args = p.parse_args(argv)

    summary = prepare(
        real_path=args.real,
        synthetic_path=args.synthetic,
        out_dir=args.out_dir,
        synthetic_ratio=args.synthetic_ratio,
        holdout_percent=args.holdout_percent,
        holdout_count=args.holdout_count,
        max_train=args.max_train,
        seed=args.seed,
        force=args.force,
        reuse_benchmark=args.reuse_benchmark,
    )
    print(
        f"prep: {summary['real']} real + {summary['synthetic_used']} synthetic "
        f"-> {summary['train']} train, {summary['benchmark']} benchmark"
    )
    print(f"  train:     {summary['train_path']}")
    print(f"  benchmark: {summary['benchmark_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
