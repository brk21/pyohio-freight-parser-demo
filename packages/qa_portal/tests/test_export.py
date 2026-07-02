"""export_training selection rules and output validity.

Uses a Django ``TestCase`` (transaction-per-test, auto rollback) against the
in-memory test DB built by conftest.py.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from freight_schema import ParsedConfirmation

from qa_portal.review.models import Confirmation

# Same lane; the two differ only in rate so we can tell which one the export
# picked (parsed vs corrected).
PARSED = [{"origin": "CHI", "destination": "LAX", "quantity": 12, "unit": "pallets", "leg": "pickup", "rate": 1847.5}]
CORRECTED = [{"origin": "CHI", "destination": "LAX", "quantity": 12, "unit": "pallets", "leg": "pickup", "rate": 2000}]


class ExportTrainingTests(TestCase):
    def _run_export(self) -> list[dict]:
        """Run export_training to a temp file and return the parsed JSONL rows."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "qa_real.jsonl"
            call_command("export_training", out=str(out))
            text = out.read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def test_selection_rules_and_label_preference(self):
        now = timezone.now()
        # A: reviewed, no correction -> exported, uses parsed.
        Confirmation.objects.create(text="A", parsed=PARSED, reviewed=now)
        # B: reviewed, corrected -> exported, uses corrected.
        Confirmation.objects.create(text="B", parsed=PARSED, corrected=CORRECTED, reviewed=now)
        # C: reviewed but excluded -> skipped.
        Confirmation.objects.create(text="C", parsed=PARSED, reviewed=now, exclude=True)
        # D: not reviewed -> skipped.
        Confirmation.objects.create(text="D", parsed=PARSED)

        rows = self._run_export()
        by_text = {row["text"]: row for row in rows}

        # Only the reviewed, non-excluded rows come through.
        assert set(by_text) == {"A", "B"}

        # Export prefers corrected (B -> rate 2000) over parsed (A -> 1847.5).
        assert json.loads(by_text["A"]["parsed_json"])[0]["rate"] == 1847.5
        assert json.loads(by_text["B"]["parsed_json"])[0]["rate"] == 2000

    def test_output_shape_and_validity(self):
        now = timezone.now()
        confirmation = Confirmation.objects.create(text="PU 12 pallets", parsed=PARSED, reviewed=now)

        rows = self._run_export()
        assert len(rows) == 1
        record = rows[0]

        # Exactly the contract's QA export keys.
        assert set(record) == {"id", "text", "parsed_json"}
        assert record["id"] == str(confirmation.pk)
        assert record["text"] == "PU 12 pallets"

        # parsed_json is a JSON *string* that re-validates through the schema.
        assert isinstance(record["parsed_json"], str)
        validated = ParsedConfirmation.model_validate_json(record["parsed_json"])
        assert validated.root[0].quantity == 12
        assert validated.root[0].unit == "pallets"

    def test_empty_when_nothing_reviewed(self):
        Confirmation.objects.create(text="unreviewed", parsed=PARSED)
        assert self._run_export() == []
