#!/usr/bin/env python
"""Django's command-line utility for the QA portal.

Run from the repo root via the workspace venv, e.g.::

    uv run python packages/qa_portal/manage.py migrate
    uv run python packages/qa_portal/manage.py seed --auto-review
    uv run python packages/qa_portal/manage.py export_training --out data/training/qa_real.jsonl
    uv run python packages/qa_portal/manage.py runserver
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "qa_portal.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:  # pragma: no cover - env sanity check
        raise ImportError(
            "Couldn't import Django. Are you running inside the uv workspace "
            "venv (e.g. `uv run python packages/qa_portal/manage.py ...`)?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
