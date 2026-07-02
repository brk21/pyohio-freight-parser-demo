"""synthetic — label-first synthetic freight-confirmation generator + DuckDB store.

Build a structured :class:`~freight_schema.ShipmentLine` first, then render messy
freetext from it, so the training label is correct by construction.
"""

from synthetic.generator import (
    MAX_COMPLEXITY,
    STYLES,
    Knobs,
    build_example,
    generate_examples,
    knobs_for,
)

__all__ = [
    "MAX_COMPLEXITY",
    "STYLES",
    "Knobs",
    "build_example",
    "generate_examples",
    "knobs_for",
]
