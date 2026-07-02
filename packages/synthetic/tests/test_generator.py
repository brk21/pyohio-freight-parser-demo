"""Tests for the label-first synthetic generator.

The invariants here are the ones the whole pipeline leans on: determinism (so the
data set is reproducible from a seed), that every gold label validates against
the shared schema, that rendered references really are literal substrings (so the
serving-time reference guard never nulls a correct one), and that multi-leg
confirmations actually produce more than one line.
"""

from __future__ import annotations

import random

import pytest

from freight_schema import ParsedConfirmation, ShipmentLine

from synthetic.generator import (
    STYLES,
    build_example,
    generate_examples,
)


def _lines(example: dict) -> list[ShipmentLine]:
    """Parse an example's gold JSON back into schema objects."""
    return ParsedConfirmation.model_validate_json(example["parsed_json"]).root


def test_generation_is_deterministic() -> None:
    assert generate_examples(50, seed=0) == generate_examples(50, seed=0)


def test_different_seeds_differ() -> None:
    # Not a hard guarantee in theory, but with this vocabulary it always holds
    # and catches an accidentally-ignored seed.
    assert generate_examples(50, seed=0) != generate_examples(50, seed=1)


def test_every_parsed_json_validates_as_shipment_lines() -> None:
    for ex in generate_examples(200, seed=7):
        lines = _lines(ex)
        assert isinstance(lines, list) and len(lines) >= 1
        for line in lines:
            assert isinstance(line, ShipmentLine)
            assert line.unit in ("pallets", "skids", "cartons", "containers")
            assert line.leg in ("pickup", "delivery")


def test_rendered_reference_is_a_literal_substring() -> None:
    for ex in generate_examples(200, seed=13):
        for line in _lines(ex):
            if line.reference is not None:
                assert line.reference in ex["text"]


def test_example_shape_and_fields() -> None:
    for ex in generate_examples(30, seed=3):
        assert set(ex) == {"id", "text", "parsed_json", "style", "complexity"}
        assert ex["style"] in STYLES
        assert isinstance(ex["complexity"], int) and ex["complexity"] >= 1
        assert isinstance(ex["text"], str) and ex["text"]


@pytest.mark.parametrize("complexity", [1, 2, 3, 4])
def test_multileg_produces_more_than_one_line(complexity: int) -> None:
    ex = build_example(random.Random(0), 0, "multileg", complexity)
    lines = _lines(ex)
    assert len(lines) > 1
    # Every leg carries the same reference (stated once, applied to all).
    refs = {line.reference for line in lines}
    assert len(refs) == 1
    ref = refs.pop()
    if ref is not None:
        # A multi-stop confirmation restates the reference on every leg.
        assert ex["text"].count(ref) >= len(lines)


def test_single_leg_styles_produce_one_line() -> None:
    for style in ("terse", "verbose", "abbrev"):
        ex = build_example(random.Random(1), 0, style, 2)
        assert len(_lines(ex)) == 1


def test_partial_dates_never_fabricate_parts() -> None:
    # If a day or year is present, the month must be too (valid prefix ladder).
    for ex in generate_examples(200, seed=21):
        for line in _lines(ex):
            if line.pickup_day is not None or line.pickup_year is not None:
                assert line.pickup_month is not None
