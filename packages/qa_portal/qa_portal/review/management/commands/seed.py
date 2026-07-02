"""``seed`` — load the hand-written seed confirmations into the review DB.

For each seed row we store the raw ``text`` and set ``parsed`` to the gold
parse. Then, to give the review cockpit something real to catch, we deliberately
corrupt a handful of records with one of each of the **four QA error classes**
the talk teaches:

    (a) hallucinated field   — a value that is *not* in the text
    (b) missing information  — a present value dropped from the parse
    (c) incorrect mapping    — the right field with the wrong value
    (d) wrong ordering       — multi-leg lines shuffled out of order

The *true* gold for every record is always known (it is the seed's ``gold``), so
a corrupted record can be "fixed" back to gold.

Two modes:

* default — corrupted records are left **unreviewed** (``reviewed=None``,
  ``corrected=None``) so a live reviewer opening the portal has genuine work to
  do. Clean records are loaded unreviewed too.
* ``--auto-review`` — simulate a *completed* QA pass: every record is stamped
  ``reviewed=now()`` and each corrupted record gets ``corrected=<gold>`` (the
  reviewer already fixed it). Ready to export as training data.

Idempotent: existing confirmations are cleared before loading.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from freight_schema.paths import seed_file

from qa_portal.review.models import Confirmation

# Types for readability: a "line" is a ShipmentLine dict; a "parse" is a list of
# them (matching freight_schema.ParsedConfirmation).
Line = dict[str, Any]
Parse = list[Line]

# A PO/BOL string designed never to appear in real confirmation text, so the
# reference guard (and a human reviewer) will flag it as hallucinated.
HALLUCINATED_REFERENCE = "PO-HALLUCINATED-000000"


class Command(BaseCommand):
    help = (
        "Load data/seed/confirmations.jsonl into the review DB, injecting the "
        "four QA error classes into a few records for the reviewer to catch."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--auto-review",
            action="store_true",
            dest="auto_review",
            help="Mark every record reviewed=now() and fix corrupted ones "
            "(corrected=gold): simulates a completed QA pass ready for export.",
        )

    def handle(self, *args, auto_review: bool = False, **options) -> None:
        path = seed_file()
        if not path.exists():
            raise CommandError(
                f"Seed file not found: {path}\n"
                "It is authored elsewhere in the repo; generate the seed data "
                "before running `seed`."
            )

        rows = self._load_rows(path)
        if not rows:
            raise CommandError(f"Seed file is empty: {path}")

        # Idempotent reload: wipe first so re-running `seed` is deterministic.
        Confirmation.objects.all().delete()

        # Decide which row demonstrates which error class (best-effort, based on
        # each row's shape). Returns {row_index: kind}.
        plan = self._plan_corruptions(rows)

        now = timezone.now()
        for i, row in enumerate(rows):
            gold: Parse = row["gold"]
            kind = plan.get(i)

            if kind is not None:
                # Corrupted: the auto-parse is wrong; gold is the true answer.
                parsed = _corrupt(kind, gold)
                corrected = copy.deepcopy(gold) if auto_review else None
            else:
                # Clean: the auto-parse already matches gold; nothing to fix.
                parsed = copy.deepcopy(gold)
                corrected = None

            Confirmation.objects.create(
                text=row["text"],
                parsed=parsed,
                corrected=corrected,
                reviewed=now if auto_review else None,
            )

        self._report(rows, plan, auto_review)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _load_rows(path) -> list[dict[str, Any]]:
        """Parse the seed JSONL into a list of ``{"id", "text", "gold"}`` dicts."""
        rows: list[dict[str, Any]] = []
        for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:  # pragma: no cover - bad seed
                raise CommandError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        return rows

    @staticmethod
    def _plan_corruptions(rows: list[dict[str, Any]]) -> dict[int, str]:
        """Assign each of the four error classes to a distinct, suitable row.

        The seed data's exact contents are owned elsewhere, so we pick rows by
        *shape* rather than by hard-coded index — the assignment is
        deterministic (first suitable row wins) and never reuses a row:

            (d) wrong ordering  -> first row with >= 2 lines (needs multi-leg)
            (b) missing info    -> first row with a droppable optional value
            (a) hallucinated    -> first remaining row with >= 1 line
            (c) incorrect map   -> first remaining row with >= 1 line
        """
        plan: dict[int, str] = {}
        used: set[int] = set()

        def claim(kind: str, predicate) -> None:
            for i, row in enumerate(rows):
                if i in used:
                    continue
                if predicate(row["gold"]):
                    plan[i] = kind
                    used.add(i)
                    return

        claim("ordering", lambda gold: len(gold) >= 2)
        claim("missing", lambda gold: any(_droppable_field(line) for line in gold))
        claim("hallucination", lambda gold: len(gold) >= 1)
        claim("mapping", lambda gold: len(gold) >= 1)
        return plan

    def _report(self, rows, plan: dict[int, str], auto_review: bool) -> None:
        """Print what got loaded and which record demonstrates which error."""
        total = len(rows)
        self.stdout.write(
            self.style.SUCCESS(
                f"Loaded {total} confirmations "
                f"({'auto-reviewed' if auto_review else 'unreviewed'})."
            )
        )
        if plan:
            self.stdout.write("Injected error classes:")
            labels = {
                "hallucination": "(a) hallucinated field  — value not in text",
                "missing": "(b) missing information — dropped a present value",
                "mapping": "(c) incorrect mapping   — right field, wrong value",
                "ordering": "(d) wrong ordering      — multi-leg lines reversed",
            }
            for i in sorted(plan):
                seed_id = rows[i].get("id", f"row {i}")
                self.stdout.write(f"  - {seed_id}: {labels[plan[i]]}")
        else:  # pragma: no cover - only if seed has no usable rows
            self.stdout.write("No rows were suitable for corruption injection.")


# ---------------------------------------------------------------------------
# The corruption functions — each returns a *new* (deep-copied) parse so the
# caller's gold is never mutated.
# ---------------------------------------------------------------------------

# Optional ShipmentLine fields we are willing to "drop" for the missing-info
# error, in preference order (most illustrative first).
_DROPPABLE = ("weight", "accessorial", "reference", "pickup_year", "pickup_month", "pickup_day")


def _droppable_field(line: Line) -> str | None:
    """Return the first present optional field we could drop from ``line``."""
    for field in _DROPPABLE:
        if line.get(field) is not None:
            return field
    return None


def _corrupt(kind: str, gold: Parse) -> Parse:
    """Return a corrupted copy of ``gold`` demonstrating error class ``kind``."""
    parsed = copy.deepcopy(gold)

    if kind == "ordering":
        # (d) wrong ordering: reverse the multi-leg lines. The set of lines is
        # correct, only their order is wrong — exactly the failure the evaluator
        # scores as an ordering error.
        parsed.reverse()

    elif kind == "missing":
        # (b) missing information: null out the first present optional value.
        for line in parsed:
            field = _droppable_field(line)
            if field is not None:
                line[field] = None
                break

    elif kind == "hallucination":
        # (a) hallucinated field: attach a reference/PO number that never
        # appears in the text. The reference guard exists precisely to catch it.
        parsed[0]["reference"] = HALLUCINATED_REFERENCE

    elif kind == "mapping":
        # (c) incorrect mapping: right field, wrong value — bump the quantity on
        # the first line so the count is plainly wrong while everything else is
        # in the correct place.
        parsed[0]["quantity"] = int(parsed[0]["quantity"]) + 7

    else:  # pragma: no cover - guarded by _plan_corruptions
        raise ValueError(f"Unknown corruption kind: {kind}")

    return parsed
