"""Shared filesystem layout for the whole pipeline.

Every stage reads and writes the same handful of locations (the seed data, the
synthetic DuckDB store, the prepared training sets, the trained adapters, the
benchmark predictions). Centralizing them here means no package hard-codes a
path that another package has to guess at.

The repo root is discovered from this file's location (the package is installed
editable in the uv workspace, so the source stays under ``packages/schema/``),
and can be overridden with ``$FREIGHT_ROOT`` for unusual setups.
"""

from __future__ import annotations

import os
from pathlib import Path

# packages/schema/freight_schema/paths.py -> parents[3] == repo root
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    """Absolute path to the repository root."""
    env = os.environ.get("FREIGHT_ROOT")
    return Path(env).resolve() if env else _DEFAULT_ROOT


def data_dir() -> Path:
    """The ``data/`` directory (created if missing)."""
    d = repo_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def seed_file() -> Path:
    """Hand-written seed confirmations + gold JSON."""
    return data_dir() / "seed" / "confirmations.jsonl"


def synthetic_db() -> Path:
    """Local DuckDB store of generated examples."""
    return data_dir() / "synthetic.duckdb"


def training_dir() -> Path:
    """Prepared (alpaca) training + benchmark JSONL."""
    d = data_dir() / "training"
    d.mkdir(parents=True, exist_ok=True)
    return d


def adapters_dir() -> Path:
    """Root of the per-model LoRA adapters (``data/adapters/<name>/``).

    Overridable with ``$FREIGHT_ADAPTERS_DIR`` (used by serving).
    """
    env = os.environ.get("FREIGHT_ADAPTERS_DIR")
    d = Path(env).resolve() if env else data_dir() / "adapters"
    d.mkdir(parents=True, exist_ok=True)
    return d


def adapter_dir(model_name: str) -> Path:
    """Directory holding the trained adapter for ``model_name`` (e.g. 'newer')."""
    return adapters_dir() / model_name


def predictions_dir() -> Path:
    """Scored benchmark prediction JSONL files (one per model)."""
    d = data_dir() / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    return d
