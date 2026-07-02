"""The one schema, imported everywhere.

Every stage of the pipeline — the QA portal, the synthetic generator, the
trainer's dataset prep, the evaluator, and the serving API — imports these
models. Changing a field happens here and only here. That single-source-of-truth
property is the whole point of the ``freight_schema`` package: the prompt the
model is trained on, the JSON the model is constrained to emit, and the records
a reviewer corrects are all the *same* Pydantic definition.

The domain is freight/logistics: carriers and freight partners send freetext
shipment confirmations ("PU 12 pallets CHI -> LAX ... rate 1,847.50"), and we
turn each into a list of structured :class:`ShipmentLine` records.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, RootModel

# ---------------------------------------------------------------------------
# The prompt contract — shared by training and serving so they never drift.
# ---------------------------------------------------------------------------

INSTRUCTION = (
    "You are an API endpoint that only speaks in JSON. "
    "You will receive shipment confirmations as raw text and output parsed JSON. "
    "'PU' means pickup; 'DLV' and 'del' mean delivery."
)


def build_prompt(text: str, guidance: str | None = None) -> str:
    """Compose the model prompt.

    Optional plain-English ``guidance`` is appended to the base instruction so
    users can steer parsing in natural language (e.g. "there are 3 shipments in
    this message"). Used identically at train time (``guidance=None``) and serve
    time, which is exactly why it lives in the shared schema package: the string
    the model saw during fine-tuning is the string it sees in production.
    """
    instruction = INSTRUCTION if not guidance else f"{INSTRUCTION}\nHint: {guidance.strip()}"
    # The trailing newline gives a stable prompt/completion boundary (the model
    # emits the JSON on the next line), which keeps train-time loss masking and
    # serve-time decoding aligned.
    return f"{instruction}\n\nConfirmation:\n{text}\n"


# ---------------------------------------------------------------------------
# The records.
# ---------------------------------------------------------------------------

# Categorical vocabularies. Kept as plain ``str`` on the model (per the schema
# shown in the talk) but documented here so the generator, the instruction, and
# reviewers agree on the allowed values.
UNITS = ("pallets", "skids", "cartons", "containers")
LEGS = ("pickup", "delivery")


class ShipmentLine(BaseModel):
    """One leg of a shipment confirmation.

    A single confirmation message may contain several of these (multi-leg), so
    the top-level parse is always a *list*. The interesting, easy-to-get-wrong
    parts of the problem are encoded in the field semantics below — the demo
    data and the evaluator deliberately exercise every one of them.
    """

    # Lane. Free-form tokens as they appear in the text: "CHI", "Dallas", "LAX".
    origin: Optional[str] = None
    destination: Optional[str] = None

    # Always present. Piece / pallet / skid count.
    quantity: int

    # Categorical: one of ``UNITS``. Always present.
    unit: str

    # Weight in lbs. May be absent -> stays null. Decimal for exact matching.
    weight: Optional[Decimal] = None

    # Partial dates: keep ONLY what the text states. "Apr 5th" -> month=4, day=5,
    # year=None. "MAR" -> month=3, day=None, year=None. Never fabricate the parts
    # the text does not give you.
    pickup_day: Optional[int] = None
    pickup_month: Optional[int] = None
    pickup_year: Optional[int] = None

    # "pickup" | "delivery". Abbreviations normalize here:
    #   PU/pu           -> "pickup"
    #   DLV/del/dlv     -> "delivery"
    leg: str

    # Fuel surcharge / detention / other accessorial. Usually null.
    accessorial: Optional[Decimal] = None

    # Always present, exact match required. The negotiated line rate.
    rate: Decimal

    # PO / BOL reference number. Typically stated once in the message and applies
    # to every line, so the generator repeats the same value on each line. The
    # serving layer applies an anti-hallucination guard (see apply_reference_guard):
    # if the predicted reference is not a literal substring of the input text, it
    # is nulled out. This mirrors a real production trick.
    reference: Optional[str] = None


class ParsedConfirmation(RootModel):
    """The public shape of a parse: a bare JSON list of :class:`ShipmentLine`."""

    root: list[ShipmentLine]


class PredictedConfirmation(BaseModel):
    """Object-with-``items`` wrapper for a parse.

    ``outlines`` (and JSON-schema-guided decoding in general) can't target a
    ``RootModel`` — a top-level array — directly, so we wrap the list of lines in
    an object. This is the *canonical* predicted type (``Decimal`` fields), used
    to validate a decoded string back into exact numeric values.
    """

    items: list[ShipmentLine]


# ---------------------------------------------------------------------------
# The constrained-decoding target.
# ---------------------------------------------------------------------------
#
# You would *expect* to hand outlines the PredictedConfirmation above and be
# done. You can't, and the reason is a great teaching moment:
#
#   pydantic renders a `Decimal` field's JSON schema with a string variant whose
#   `pattern` uses a negative lookahead — `^(?!^[-+.]*$)[+-]?0*\d*\.?\d*$`.
#   outlines compiles JSON schemas to a finite-state automaton over the token
#   vocabulary, and a DFA cannot express lookahead, so the build fails outright
#   with "Failed to build DFA / error building NFA".
#
# The fix is to decode against a *mirror* schema that swaps `Decimal` for `float`
# (a plain `{"type": "number"}`, DFA-friendly) and pins the categorical fields to
# `Literal`s (which both shrinks the automaton and guarantees a valid unit/leg).
# We then re-validate the raw decoded JSON *text* into the canonical Decimal
# schema — pydantic reads each JSON number token straight into a `Decimal`
# without ever routing through float, so exact numeric matching is preserved.
#
# So: decode with DecodeConfirmation (fast, always valid) -> get raw JSON string
# -> decode_to_confirmation() -> ParsedConfirmation with exact Decimals.


class DecodeLine(BaseModel):
    """DFA-friendly mirror of :class:`ShipmentLine` used only for decoding.

    Identical field names and meanings, but ``Decimal`` -> ``float`` and the
    categoricals are ``Literal``s. Never stored or compared against; it exists
    purely to shape the constrained-decoding automaton.
    """

    origin: Optional[str] = None
    destination: Optional[str] = None
    quantity: int
    unit: Literal["pallets", "skids", "cartons", "containers"]
    weight: Optional[float] = None
    pickup_day: Optional[int] = None
    pickup_month: Optional[int] = None
    pickup_year: Optional[int] = None
    leg: Literal["pickup", "delivery"]
    accessorial: Optional[float] = None
    rate: float
    reference: Optional[str] = None


class DecodeConfirmation(BaseModel):
    """The object handed to ``outlines.Generator`` as the output type."""

    items: list[DecodeLine]


def decode_to_confirmation(raw_json: str) -> ParsedConfirmation:
    """Turn a raw decoded ``DecodeConfirmation`` JSON string into the canonical
    :class:`ParsedConfirmation` (exact ``Decimal`` values).

    ``raw_json`` is the string outlines returns: ``{"items": [ ... ]}``. We
    validate it straight into :class:`PredictedConfirmation`, whose ``items`` are
    ``Decimal``-typed :class:`ShipmentLine`s — pydantic parses the JSON number
    tokens losslessly — then unwrap to the public list form.
    """
    predicted = PredictedConfirmation.model_validate_json(raw_json)
    return ParsedConfirmation(root=predicted.items)


# ---------------------------------------------------------------------------
# Shared helpers used across stages (so the behavior is defined exactly once).
# ---------------------------------------------------------------------------


def apply_reference_guard(items: list[ShipmentLine], text: str) -> list[ShipmentLine]:
    """Null out any ``reference`` the model invented.

    Anti-hallucination guard from production: a reference (PO/BOL number) is only
    trustworthy if it appears verbatim in the source text. If the model emits a
    reference that is not a literal substring of ``text``, we drop it to ``None``
    rather than pass a fabricated identifier downstream. Returns the same list
    (mutated in place) for convenience.
    """
    for line in items:
        if line.reference is not None and line.reference not in text:
            line.reference = None
    return items


def _json_default(obj: object) -> object:
    """JSON encoder hook: render ``Decimal`` as a JSON *number*, not a string.

    We keep values as :class:`~decimal.Decimal` in memory for exact comparison,
    but emit them as plain JSON numbers so the training data and API responses
    read naturally (``"rate": 1847.5``, not ``"rate": "1847.50"``). Integral
    values collapse to ``int``; the rest to ``float``. Our money/weight values
    are clean two-decimal quantities that round-trip through this exactly, and
    the comparison path in the evaluator re-parses the JSON *number token*
    straight back into ``Decimal`` (lossless — pydantic never routes through
    float), so precision is preserved where it matters.
    """
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dump_lines(lines: list[ShipmentLine]) -> str:
    """Serialize a list of lines to canonical JSON text.

    Numbers stay numbers (see :func:`_json_default`), and we drop fields that are
    ``None`` (absent). The always-present fields (quantity, unit, leg, rate) are
    never null, so a single-leg line collapses to just the values the text
    actually stated. This keeps the JSON readable, keeps training targets short
    (faster to generate on CPU), and — because every optional field defaults to
    ``None`` when re-parsed — round-trips losslessly. Absent == null everywhere.
    """
    compact = [
        {k: v for k, v in line.model_dump().items() if v is not None} for line in lines
    ]
    return json.dumps(compact, default=_json_default)


def dump_confirmation(confirmation: ParsedConfirmation) -> str:
    """Serialize a :class:`ParsedConfirmation` to canonical JSON text."""
    return dump_lines(confirmation.root)
