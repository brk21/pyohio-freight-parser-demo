"""freight_schema — the single source of truth for the freight-parser pipeline.

Import the models and the prompt contract from here everywhere else in the repo.
"""

from freight_schema.models import (
    INSTRUCTION,
    LEGS,
    UNITS,
    DecodeConfirmation,
    DecodeLine,
    ParsedConfirmation,
    PredictedConfirmation,
    ShipmentLine,
    apply_reference_guard,
    build_prompt,
    decode_to_confirmation,
    dump_confirmation,
    dump_lines,
)

__all__ = [
    "INSTRUCTION",
    "LEGS",
    "UNITS",
    "DecodeConfirmation",
    "DecodeLine",
    "ParsedConfirmation",
    "PredictedConfirmation",
    "ShipmentLine",
    "apply_reference_guard",
    "build_prompt",
    "decode_to_confirmation",
    "dump_confirmation",
    "dump_lines",
]
