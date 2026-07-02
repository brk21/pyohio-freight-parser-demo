"""Fine-tune the older and newer models in one command.

This is what powers the talk's headline comparison: same data, same recipe, two
base generations. ``make demo`` calls this so the benchmark can show the
generational gap.

Run:  uv run python -m finetune.train_all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from finetune.train import train_model

# The two models the demo compares. "lightweight" is available via
# `train --model lightweight` but is not part of the default older-vs-newer story.
DEFAULT_MODELS = ["older", "newer"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="Registry names to train (default: older newer).")
    p.add_argument("--train-file", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    for name in args.models:
        train_model(
            name=name,
            train_file=args.train_file,
            epochs=args.epochs,
            max_train=args.max_train,
            seed=args.seed,
        )
    print(f"train_all: trained {', '.join(args.models)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
