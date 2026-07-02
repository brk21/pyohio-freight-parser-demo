"""Tests for the field-level scorer precedence."""

from eval.score import CATEGORIES, categorize, score_predictions


def _e():
    return [{
        "quantity": 12, "unit": "pallets", "leg": "pickup", "rate": 1847.5,
        "weight": 4200, "origin": "CHI", "destination": "LAX",
        "pickup_month": 3, "pickup_day": 14, "reference": "PO-1",
    }]


def _mut(**kw):
    row = dict(_e()[0])
    row.update(kw)
    return [row]


def test_exact_match_is_correct():
    assert categorize(_e(), _mut()) == "CORRECT"


def test_trailing_zero_still_correct():
    assert categorize(_e(), _mut(rate=1847.50)) == "CORRECT"


def test_error_when_prediction_missing():
    assert categorize(_e(), None, error="boom") == "ERROR"
    assert categorize(_e(), None) == "ERROR"


def test_count_mismatches():
    assert categorize(_e(), _e() + _e()) == "EXTRA_ITEMS"
    assert categorize(_e(), []) == "MISSING_ITEMS"


def test_unordered_multiset():
    a = {"quantity": 1, "unit": "pallets", "leg": "pickup", "rate": 10}
    b = {"quantity": 2, "unit": "skids", "leg": "delivery", "rate": 20}
    assert categorize([a, b], [b, a]) == "UNORDERED"


def test_field_precedence():
    # quantity outranks rate
    assert categorize(_e(), _mut(quantity=5, rate=1.0)) == "BAD_QUANTITY"
    assert categorize(_e(), _mut(rate=1.0)) == "BAD_RATE"
    assert categorize(_e(), _mut(reference="PO-9")) == "BAD_REFERENCE"
    assert categorize(_e(), _mut(pickup_day=15)) == "BAD_DATE"
    assert categorize(_e(), _mut(weight=1)) == "BAD_WEIGHT"
    assert categorize(_e(), _mut(unit="skids")) == "BAD_UNIT"
    assert categorize(_e(), _mut(leg="delivery")) == "BAD_LEG"


def test_other_for_uncategorized_field():
    # origin differs only -> not a categorized field -> OTHER
    assert categorize(_e(), _mut(origin="DAL")) == "OTHER"


def test_score_predictions_adds_category():
    rows = [{"expected": _e(), "predicted": _mut(), "error": None}]
    scored = score_predictions(rows)
    assert scored[0]["category"] == "CORRECT"
    assert set(scored[0]).issuperset({"expected", "predicted", "category"})


def test_all_categories_reachable_are_valid():
    assert "OTHER" in CATEGORIES and "CORRECT" in CATEGORIES
