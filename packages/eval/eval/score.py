"""Field-level scoring: turn each prediction into one debuggable category.

One accuracy number hides *where* a model fails. Instead we bucket every
prediction into exactly one mutually-exclusive category, chosen by a fixed
precedence, so the benchmark points straight at the field that broke.

Precedence (first match wins):

    ERROR > CORRECT > UNORDERED > EXTRA_ITEMS > MISSING_ITEMS
          > BAD_QUANTITY > BAD_REFERENCE > BAD_DATE > BAD_WEIGHT
          > BAD_UNIT > BAD_LEG > BAD_RATE > OTHER

- ``CORRECT``   — exact match (Decimal-aware for rate/weight/accessorial).
- ``UNORDERED`` — same multiset of lines, different order (counts as accurate).
- field buckets — compared line-by-line, but only when the line counts match.
- ``OTHER``     — counts match, differs only in an uncategorized field
                  (origin/destination/accessorial).
"""

from __future__ import annotations

from freight_schema import ShipmentLine

# Full category vocabulary, in precedence order (used to order report output too).
CATEGORIES = [
    "ERROR", "CORRECT", "UNORDERED", "EXTRA_ITEMS", "MISSING_ITEMS",
    "BAD_QUANTITY", "BAD_REFERENCE", "BAD_DATE", "BAD_WEIGHT", "BAD_UNIT",
    "BAD_LEG", "BAD_RATE", "OTHER",
]
ACCURATE = {"CORRECT", "UNORDERED"}

# Field -> category, in the precedence order used once line counts match.
_FIELD_CATEGORIES: list[tuple[str, str]] = [
    ("quantity", "BAD_QUANTITY"),
    ("reference", "BAD_REFERENCE"),
    ("date", "BAD_DATE"),          # composite: pickup_day/month/year
    ("weight", "BAD_WEIGHT"),
    ("unit", "BAD_UNIT"),
    ("leg", "BAD_LEG"),
    ("rate", "BAD_RATE"),
]


def _to_lines(objs: list[dict]) -> list[ShipmentLine]:
    # pydantic coerces JSON numbers into exact Decimals (verified: 2.16 stays 2.16).
    return [ShipmentLine(**o) for o in objs]


def _same_multiset(expected: list[ShipmentLine], predicted: list[ShipmentLine]) -> bool:
    """True if ``predicted`` is a reordering of ``expected`` (value equality)."""
    if len(expected) != len(predicted):
        return False
    remaining = list(predicted)
    for e in expected:
        for i, p in enumerate(remaining):
            if e == p:  # pydantic model equality is value-based (Decimal-aware)
                remaining.pop(i)
                break
        else:
            return False
    return True


def _field_differs(field: str, a: ShipmentLine, b: ShipmentLine) -> bool:
    if field == "date":
        return (a.pickup_day, a.pickup_month, a.pickup_year) != (
            b.pickup_day, b.pickup_month, b.pickup_year
        )
    return getattr(a, field) != getattr(b, field)


def categorize(
    expected: list[dict],
    predicted: list[dict] | None,
    error: str | None = None,
) -> str:
    """Assign one category to a prediction. See module docstring for precedence."""
    if error or predicted is None:
        return "ERROR"
    try:
        exp = _to_lines(expected)
        pred = _to_lines(predicted)
    except Exception:
        return "ERROR"

    if len(exp) == len(pred) and all(a == b for a, b in zip(exp, pred)):
        return "CORRECT"
    if _same_multiset(exp, pred):
        return "UNORDERED"
    if len(pred) > len(exp):
        return "EXTRA_ITEMS"
    if len(pred) < len(exp):
        return "MISSING_ITEMS"

    # Counts match but it isn't correct: find the highest-precedence field that
    # differs on any aligned line pair.
    for field, category in _FIELD_CATEGORIES:
        if any(_field_differs(field, a, b) for a, b in zip(exp, pred)):
            return category
    return "OTHER"


# The full field set, for the field-level accuracy metric.
_FIELDS = list(ShipmentLine.model_fields)


def field_accuracy(
    expected: list[dict],
    predicted: list[dict] | None,
    error: str | None = None,
) -> float:
    """Fraction of individual fields the prediction got right (0..1).

    Exact-match ``CORRECT`` is an unforgiving headline — a single wrong field
    sinks the whole record — so small models can look uniformly bad on it even
    when one is clearly better. This finer metric counts matching fields across
    aligned lines (extra/missing lines count all-wrong), so it surfaces the
    generational gap the categories hint at. It complements, and never replaces,
    the exact-match gate.
    """
    if error or predicted is None:
        return 0.0
    try:
        exp = _to_lines(expected)
        pred = _to_lines(predicted)
    except Exception:
        return 0.0
    n_lines = max(len(exp), len(pred))
    if n_lines == 0:
        return 1.0
    total = n_lines * len(_FIELDS)
    matches = 0
    for a, b in zip(exp, pred):  # only aligned lines can score; the rest are misses
        matches += sum(1 for f in _FIELDS if getattr(a, f) == getattr(b, f))
    return matches / total


def score_predictions(rows: list[dict]) -> list[dict]:
    """Add a ``category`` and a ``field_accuracy`` to each raw prediction row."""
    scored = []
    for row in rows:
        expected = row.get("expected", [])
        predicted = row.get("predicted")
        error = row.get("error")
        scored.append({
            **row,
            "category": categorize(expected, predicted, error),
            "field_accuracy": round(field_accuracy(expected, predicted, error), 4),
        })
    return scored
