"""Schema validation in Confirmation.clean().

These are pure unit tests: constructing a model instance and calling clean()
never touches the database, so no DB fixture is needed.
"""

from __future__ import annotations

import pytest
from django.core.exceptions import ValidationError

from qa_portal.review.models import Confirmation

# A valid parse: one pickup leg with the required quantity/unit/leg/rate.
GOOD_PARSE = [
    {
        "origin": "CHI",
        "destination": "LAX",
        "quantity": 12,
        "unit": "pallets",
        "leg": "pickup",
        "rate": 1847.50,
    }
]


def test_clean_accepts_valid_parsed():
    """A schema-valid ``parsed`` passes clean() without raising."""
    confirmation = Confirmation(text="PU 12 pallets CHI->LAX rate 1847.50", parsed=GOOD_PARSE)
    confirmation.clean()  # must not raise


def test_clean_accepts_valid_corrected():
    """A valid ``corrected`` alongside a valid ``parsed`` also passes."""
    confirmation = Confirmation(
        text="x",
        parsed=GOOD_PARSE,
        corrected=[{"quantity": 3, "unit": "skids", "leg": "delivery", "rate": 500}],
    )
    confirmation.clean()  # must not raise


def test_clean_rejects_bad_parsed_missing_required_fields():
    """Dropping required fields (quantity, rate) makes ``parsed`` invalid."""
    bad = [{"unit": "pallets", "leg": "pickup"}]  # no quantity, no rate
    confirmation = Confirmation(text="junk", parsed=bad)
    with pytest.raises(ValidationError) as exc_info:
        confirmation.clean()
    assert "parsed" in exc_info.value.message_dict


def test_clean_rejects_non_list_parsed():
    """``parsed`` must be a *list* of lines; an object is rejected."""
    confirmation = Confirmation(text="junk", parsed={"not": "a list"})
    with pytest.raises(ValidationError):
        confirmation.clean()


def test_clean_rejects_bad_corrected():
    """A valid ``parsed`` does not excuse an invalid ``corrected``."""
    confirmation = Confirmation(
        text="x",
        parsed=GOOD_PARSE,
        corrected=[{"unit": "pallets", "leg": "pickup"}],  # missing quantity + rate
    )
    with pytest.raises(ValidationError) as exc_info:
        confirmation.clean()
    assert "corrected" in exc_info.value.message_dict


def test_clean_ignores_absent_corrected():
    """A null ``corrected`` is fine — only ``parsed`` is validated."""
    confirmation = Confirmation(text="x", parsed=GOOD_PARSE, corrected=None)
    confirmation.clean()  # must not raise


def test_training_label_prefers_corrected():
    """training_label returns corrected when set, else parsed."""
    corrected = [{"quantity": 1, "unit": "cartons", "leg": "delivery", "rate": 42}]
    with_fix = Confirmation(text="x", parsed=GOOD_PARSE, corrected=corrected)
    without_fix = Confirmation(text="x", parsed=GOOD_PARSE)
    assert with_fix.training_label == corrected
    assert without_fix.training_label == GOOD_PARSE
