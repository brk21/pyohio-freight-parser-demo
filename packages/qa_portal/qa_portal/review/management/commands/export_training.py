"""``export_training`` — emit reviewed confirmations as training JSONL.

Turns the review DB into the QA export format the rest of the pipeline expects
(see the build contract)::

    {"id": str, "text": str, "parsed_json": "<json string of the gold list>"}

Rules, all enforced here so the exported set is exactly the "trusted labels":

* only rows a reviewer has signed off on (``reviewed`` is set) are exported;
* rows flagged ``exclude`` are skipped (junk kept out of training);
* the label is the reviewer's ``corrected`` when present, else ``parsed``
  (:pyattr:`Confirmation.training_label`);
* every label is re-validated through the shared schema and serialized with
  :func:`freight_schema.dump_lines`, so ``parsed_json`` is canonical and valid.

Default output is ``data/training/qa_real.jsonl`` (via
:func:`freight_schema.paths.training_dir`).
"""

from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from freight_schema import ParsedConfirmation, dump_lines
from freight_schema.paths import training_dir

from qa_portal.review.models import Confirmation


class Command(BaseCommand):
    help = "Export reviewed, non-excluded confirmations as training JSONL."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--out",
            default=None,
            help="Output JSONL path (default: data/training/qa_real.jsonl).",
        )

    def handle(self, *args, out: str | None = None, **options) -> None:
        out_path = Path(out) if out else training_dir() / "qa_real.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Only reviewed, non-excluded rows are trusted training labels.
        queryset = (
            Confirmation.objects.filter(reviewed__isnull=False, exclude=False)
            .order_by("pk")
        )

        written = 0
        skipped = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for confirmation in queryset:
                parsed_json = self._canonical_parsed_json(confirmation)
                if parsed_json is None:
                    # An invalid label should never reach export (clean() guards
                    # form saves), but skip-and-warn rather than abort the run.
                    skipped += 1
                    self.stderr.write(
                        f"Skipping #{confirmation.pk}: label failed schema validation."
                    )
                    continue
                record = {
                    "id": str(confirmation.pk),
                    "text": confirmation.text,
                    "parsed_json": parsed_json,
                }
                fh.write(json.dumps(record) + "\n")
                written += 1

        summary = f"Wrote {written} training examples to {out_path}"
        if skipped:
            summary += f" ({skipped} skipped as invalid)"
        self.stdout.write(self.style.SUCCESS(summary))

    @staticmethod
    def _canonical_parsed_json(confirmation: Confirmation) -> str | None:
        """Validate the row's label and render it as canonical parsed_json.

        Returns ``None`` if the stored label is not a valid parse.
        """
        try:
            lines = ParsedConfirmation.model_validate(confirmation.training_label).root
        except Exception:  # pydantic ValidationError / type error
            return None
        # dump_lines emits numbers as numbers with Decimals preserved — the same
        # canonical form every other stage reads.
        return dump_lines(lines)
