"""Tests for the shared schema: validation, exact numbers, prompt, guards."""

from decimal import Decimal

from freight_schema import (
    ParsedConfirmation,
    ShipmentLine,
    apply_reference_guard,
    build_prompt,
    decode_to_confirmation,
    dump_lines,
    INSTRUCTION,
)


def _line(**kw):
    base = dict(quantity=1, unit="pallets", leg="pickup", rate=Decimal("100.00"))
    base.update(kw)
    return ShipmentLine(**base)


def test_required_and_optional_fields():
    line = _line()
    assert line.weight is None and line.reference is None and line.pickup_month is None
    # required fields present
    assert line.quantity == 1 and line.unit == "pallets" and line.rate == Decimal("100")


def test_exact_numeric_matching_via_json_number_token():
    # The tricky non-binary-exact rate must survive JSON round-trip as an exact Decimal.
    raw = '{"items":[{"quantity":3,"unit":"skids","leg":"delivery","rate":2.16}]}'
    conf = decode_to_confirmation(raw)
    assert conf.root[0].rate == Decimal("2.16")
    # trailing zeros compare equal
    assert _line(rate=Decimal("1847.50")) == _line(rate=Decimal("1847.5"))


def test_dump_lines_emits_numbers_not_strings():
    text = dump_lines([_line(rate=Decimal("1847.50"), weight=Decimal("4200"))])
    assert '"rate": 1847.5' in text
    assert '"weight": 4200' in text  # integral -> int
    # and it round-trips back into the canonical Decimal schema
    conf = ParsedConfirmation.model_validate_json('{"root": ' + text + "}") if False else None
    assert conf is None  # (RootModel validated below instead)
    reparsed = decode_to_confirmation('{"items": ' + text + "}")
    assert reparsed.root[0].rate == Decimal("1847.5")


def test_reference_guard_nulls_hallucinated_reference():
    lines = [_line(reference="PO-999")]
    apply_reference_guard(lines, "no such reference in here")
    assert lines[0].reference is None
    # but a real substring survives
    lines = [_line(reference="PO-123")]
    apply_reference_guard(lines, "order PO-123 confirmed")
    assert lines[0].reference == "PO-123"


def test_build_prompt_appends_guidance():
    assert build_prompt("hello") == f"{INSTRUCTION}\n\nConfirmation:\nhello\n"
    withhint = build_prompt("hello", "there are 3 shipments")
    assert "Hint: there are 3 shipments" in withhint
    assert "Confirmation:\nhello" in withhint


def test_parsed_confirmation_validates_list():
    conf = ParsedConfirmation.model_validate([_line().model_dump()])
    assert len(conf.root) == 1
