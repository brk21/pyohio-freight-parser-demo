"""Synthetic freight-confirmation generator.

The whole trick of this module is **label-first generation**: we build a fully
structured :class:`~freight_schema.ShipmentLine` (or a list of them) by sampling
from a generic freight vocabulary, and only *then* render messy freetext *from*
that structure. Because the text is derived from the record — never the other
way round — the gold label is correct **by construction**. There is no parsing,
no heuristics, and no risk of the text and the label disagreeing.

Everything is a pure function of a seed: :func:`generate_examples` threads a
single ``random.Random(seed)`` through the whole run, so the same seed always
produces byte-identical output. That determinism is what lets the demo, the
tests, and the training pipeline agree on exactly which examples exist.

Render **styles** capture the ways a real confirmation arrives:

* ``terse``    – shorthand a dispatcher taps out: ``PU 12 plt CHI-LAX 1,847.50``
* ``verbose``  – a full-sentence confirmation email
* ``abbrev``   – abbreviation-heavy, label-tagged: ``PU 12 PLT ORIG DFW ... RT``
* ``multileg`` – a multi-stop load confirmation; every leg repeats the reference

Complexity **knobs** (see :class:`Knobs`) scale the difficulty: number of legs,
how often optional fields go missing, how dense the abbreviations are, and how
partial the dates get.

Nothing here is real: the carrier names, lane codes, and commodities are all
invented, generic freight tokens.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

from freight_schema import UNITS, ShipmentLine, dump_lines

# ---------------------------------------------------------------------------
# Generic, invented freight vocabulary. Nothing here names a real company,
# person, or place beyond ordinary US city names / airport-style codes.
# ---------------------------------------------------------------------------

CARRIERS: tuple[str, ...] = (
    "Ridgeline Freight",
    "Cascade Logistics",
    "Beacon Carriers",
    "Ironwood Transport",
    "Northwind Freightways",
    "Summit Drayage",
    "Harbor Point Logistics",
    "Meridian Haulage",
)

# Each lane token is a (code, city) pair; the renderer picks whichever surface
# form fits the style, and we store *that same string* on the record so the
# label matches the text exactly.
PLACES: tuple[tuple[str, str], ...] = (
    ("CHI", "Chicago"),
    ("LAX", "Los Angeles"),
    ("DFW", "Dallas"),
    ("ATL", "Atlanta"),
    ("SEA", "Seattle"),
    ("DEN", "Denver"),
    ("PHX", "Phoenix"),
    ("MEM", "Memphis"),
    ("CLT", "Charlotte"),
    ("STL", "St. Louis"),
)

# Flavor text only — the schema has no commodity field. Real confirmations are
# full of noise like this, so we sprinkle it into verbose renders.
COMMODITIES: tuple[str, ...] = (
    "palletized dry goods",
    "auto parts",
    "canned goods",
    "HVAC units",
    "building materials",
    "packaged beverages",
    "industrial fasteners",
    "mixed retail freight",
)

# PO / BOL / PRO reference prefixes. A reference is stored verbatim as the exact
# token rendered into the text, guaranteeing it is a literal substring.
REF_PREFIXES: tuple[str, ...] = ("PO", "BOL", "REF", "SO", "PRO", "LOAD")

# Surface forms for each unit. The full word is always the schema value; the
# abbreviations are what a hurried dispatcher actually types.
_UNIT_ABBREV: dict[str, tuple[str, ...]] = {
    "pallets": ("plt", "plts", "pal", "pals"),
    "skids": ("skd", "skds", "sk"),
    "cartons": ("ctn", "ctns", "cs", "cases"),
    "containers": ("cntr", "cntrs", "cont", "box"),
}

# (month_number, abbrev, full_name)
_MONTHS: tuple[tuple[int, str, str], ...] = (
    (1, "Jan", "January"),
    (2, "Feb", "February"),
    (3, "Mar", "March"),
    (4, "Apr", "April"),
    (5, "May", "May"),
    (6, "Jun", "June"),
    (7, "Jul", "July"),
    (8, "Aug", "August"),
    (9, "Sep", "September"),
    (10, "Oct", "October"),
    (11, "Nov", "November"),
    (12, "Dec", "December"),
)

STYLES: tuple[str, ...] = ("terse", "verbose", "abbrev", "multileg")

# Complexity levels 1..MAX_COMPLEXITY scale the knobs below. A single integer is
# stored per example (the ``complexity`` column) so the difficulty of the data
# is queryable.
MAX_COMPLEXITY = 4


# ---------------------------------------------------------------------------
# Complexity knobs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Knobs:
    """The dials that turn a style + complexity level into concrete difficulty.

    * ``n_legs``           – how many :class:`ShipmentLine` records the message
      contains (always 1 except for the ``multileg`` style).
    * ``missing_prob``     – probability that any given *optional* field
      (weight, lane, date, reference, accessorial) is omitted.
    * ``abbrev_density``   – 0..1 chance of choosing an abbreviated surface form
      over a spelled-out one (leg keyword, unit, lane connector, ...).
    * ``partial_date_prob``– how aggressively dates are truncated to only some
      of month / day / year.
    """

    n_legs: int
    missing_prob: float
    abbrev_density: float
    partial_date_prob: float


def knobs_for(style: str, complexity: int, rng: random.Random) -> Knobs:
    """Derive concrete :class:`Knobs` from a style and a complexity level.

    Each style has a characteristic baseline (verbose is clean and complete;
    abbrev is dense and lossy), and the complexity level nudges every dial toward
    "harder" — more missing fields, more truncated dates, and (for multileg) more
    stops. ``rng`` is only consumed to pick the leg count so the whole run stays
    deterministic.
    """
    # Real confirmations vary a lot: plenty are terse, stating only the lane,
    # count, unit, and rate. These baselines keep a healthy share of short/simple
    # messages (which small models can actually get exactly right) alongside the
    # harder ones with weights, dates, and references.
    base = {
        "terse": dict(missing=0.50, abbrev=0.45, pdate=0.60),
        "verbose": dict(missing=0.25, abbrev=0.0, pdate=0.30),
        "abbrev": dict(missing=0.45, abbrev=0.90, pdate=0.60),
        "multileg": dict(missing=0.35, abbrev=0.50, pdate=0.50),
    }[style]

    scale = 0.05 * (complexity - 1)
    missing = min(0.55, base["missing"] + scale)
    pdate = min(0.85, base["pdate"] + scale)

    if style == "multileg":
        # 2..4 stops, more at higher complexity. Always >= 2 so the label really
        # is multi-line.
        n_legs = rng.randint(2, 2 + min(complexity, 2))
    else:
        n_legs = 1

    return Knobs(
        n_legs=n_legs,
        missing_prob=missing,
        abbrev_density=base["abbrev"],
        partial_date_prob=pdate,
    )


# ---------------------------------------------------------------------------
# Small value samplers. Each returns a schema-ready value.
# ---------------------------------------------------------------------------


def _make_rate(rng: random.Random) -> Decimal:
    """A negotiated line rate, e.g. Decimal('1847.50'). Always present."""
    dollars = rng.randint(350, 4800)
    cents = rng.choice((0, 25, 50, 75, 95, 99))
    return Decimal(f"{dollars}.{cents:02d}")


def _make_weight(rng: random.Random) -> Decimal:
    """A whole-hundreds weight in lbs, e.g. Decimal('15400')."""
    return Decimal(rng.randint(5, 420) * 100)


def _make_accessorial(rng: random.Random) -> Decimal:
    """A fuel/detention/lumper accessorial charge. Usually absent (see caller)."""
    return Decimal(f"{rng.choice((75, 100, 125, 150, 200, 225, 250, 300))}.00")


def _make_reference(rng: random.Random) -> str:
    """A PO/BOL/PRO token stored *verbatim* — this exact string is rendered."""
    prefix = rng.choice(REF_PREFIXES)
    sep = rng.choice(("-", " ", "#", ""))
    number = rng.randint(1000, 99999)
    return f"{prefix}{sep}{number}"


def _make_lane(rng: random.Random, style: str) -> tuple[str, str]:
    """Pick two distinct places and the surface form appropriate to the style.

    We commit to the surface (code vs. city) here and store it on the record, so
    ``origin``/``destination`` are literal substrings of the rendered text.
    """
    a, b = rng.sample(PLACES, 2)
    if style == "verbose":
        use_city = True
    elif style in ("terse", "abbrev"):
        use_city = False
    else:  # multileg: a realistic mix
        use_city = rng.random() < 0.4
    idx = 1 if use_city else 0
    return a[idx], b[idx]


def _make_date_parts(
    rng: random.Random, knobs: Knobs
) -> tuple[int | None, int | None, int | None]:
    """Sample a (day, month, year) triple, truncated per the partial-date knob.

    We only ever populate a valid *prefix* of the ladder month -> +day -> +year,
    and we NEVER fabricate a part we did not decide to state. The renderer will
    show exactly the parts that are non-null.
    """
    if rng.random() < knobs.missing_prob:
        return None, None, None  # no date at all

    month = rng.randint(1, 12)
    r = rng.random()
    if r < knobs.partial_date_prob * 0.5:
        return None, month, None  # month only ("MAR")
    if r < knobs.partial_date_prob:
        return rng.randint(1, 28), month, None  # month + day ("Apr 5")
    return rng.randint(1, 28), month, rng.choice((2025, 2026))  # full date


def _build_line(
    rng: random.Random, leg: str, style: str, knobs: Knobs, reference: str | None
) -> ShipmentLine:
    """Assemble one fully-structured shipment line by sampling fields."""
    quantity = rng.randint(1, 40)
    unit = rng.choice(UNITS)
    rate = _make_rate(rng)

    # Lane: usually present. Occasionally origin-only (dropped destination).
    origin: str | None = None
    destination: str | None = None
    if rng.random() > knobs.missing_prob * 0.5:
        origin, destination = _make_lane(rng, style)
        if rng.random() < knobs.missing_prob * 0.3:
            destination = None

    weight = _make_weight(rng) if rng.random() > knobs.missing_prob else None

    # Dates describe the *pickup*, and the schema only has pickup_* fields, so we
    # only attach a date to pickup legs. Delivery legs stay date-less.
    if leg == "pickup":
        pickup_day, pickup_month, pickup_year = _make_date_parts(rng, knobs)
    else:
        pickup_day = pickup_month = pickup_year = None

    # Accessorials are rare in real confirmations; keep them so.
    accessorial = _make_accessorial(rng) if rng.random() < 0.18 else None

    return ShipmentLine(
        origin=origin,
        destination=destination,
        quantity=quantity,
        unit=unit,
        weight=weight,
        pickup_day=pickup_day,
        pickup_month=pickup_month,
        pickup_year=pickup_year,
        leg=leg,
        accessorial=accessorial,
        rate=rate,
        reference=reference,
    )


# ---------------------------------------------------------------------------
# Surface-form helpers. Each turns a stored value into messy text, with the
# abbreviation density deciding how terse the surface gets.
# ---------------------------------------------------------------------------


def _pick(rng: random.Random, clean: list[str], abbrev: list[str], density: float) -> str:
    """Choose an abbreviated form with probability ``density``, else a clean one."""
    return rng.choice(abbrev if rng.random() < density else clean)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _leg_word(rng: random.Random, leg: str, density: float) -> str:
    """Render the pickup/delivery keyword. All forms normalize back to the leg.

    ``PU``/``pu`` -> pickup; ``DLV``/``del``/``dlv`` -> delivery (see INSTRUCTION).
    """
    if leg == "pickup":
        clean = ["Pickup", "pickup", "pick up", "PU"]
        abbrev = ["PU", "P/U", "pu"]
    else:
        clean = ["Delivery", "delivery", "deliver to", "DLV"]
        abbrev = ["DLV", "DEL", "dlv", "del"]
    return _pick(rng, clean, abbrev, density)


def _unit_word(rng: random.Random, unit: str, density: float) -> str:
    return _pick(rng, [unit], list(_UNIT_ABBREV[unit]), density)


def _lane_phrase(
    rng: random.Random, origin: str | None, destination: str | None, density: float
) -> str | None:
    if origin is None and destination is None:
        return None
    if origin is not None and destination is not None:
        clean = [f"{origin} to {destination}", f"from {origin} to {destination}"]
        abbrev = [
            f"{origin}->{destination}",
            f"{origin}-{destination}",
            f"{origin} > {destination}",
            f"ORIG {origin} DEST {destination}",
        ]
        return _pick(rng, clean, abbrev, density)
    if origin is not None:
        return _pick(rng, [f"from {origin}", f"out of {origin}"], [f"orig {origin}", f"@ {origin}"], density)
    return _pick(rng, [f"to {destination}", f"delivering to {destination}"], [f"dest {destination}", f"-> {destination}"], density)


def _weight_phrase(rng: random.Random, weight: Decimal, density: float) -> str:
    w = int(weight)
    clean = [f"{w:,} lbs", f"{w:,} pounds", f"weighs {w:,} lbs"]
    abbrev = [f"{w}#", f"{w:,}#", f"WT {w:,}#", f"{w} lb"]
    return _pick(rng, clean, abbrev, density)


def _rate_phrase(rng: random.Random, rate: Decimal, density: float) -> str:
    r = f"{rate:,.2f}"
    clean = [f"${r}", f"rate ${r}", f"linehaul ${r}", f"at ${r}"]
    abbrev = [f"RT {r}", f"${r}", f"@${r}", f"@ ${r}"]
    return _pick(rng, clean, abbrev, density)


def _accessorial_phrase(rng: random.Random, accessorial: Decimal, density: float) -> str:
    a = f"{accessorial:,.2f}"
    kind = rng.choice(("fuel surcharge", "detention", "lumper", "FSC", "layover"))
    return _pick(rng, [f"{kind} ${a}"], [f"{kind} {a}", f"+${a} {kind}"], density)


def _date_phrase(rng: random.Random, line: ShipmentLine, density: float) -> str | None:
    """Render exactly the date parts that are stored (never more)."""
    m = line.pickup_month
    if m is None:
        return None
    _, abbr, full = _MONTHS[m - 1]
    d, y = line.pickup_day, line.pickup_year

    if d is None:  # month only
        return _pick(rng, [full, f"in {full}"], [abbr.upper(), abbr, f"mo {abbr}"], density)
    if y is None:  # month + day
        return _pick(
            rng,
            [f"{full} {_ordinal(d)}", f"{full} {d}"],
            [f"{abbr} {d}", f"{m}/{d}", f"{abbr} {_ordinal(d)}"],
            density,
        )
    # full date
    return _pick(
        rng,
        [f"{full} {d}, {y}", f"{full} {d} {y}"],
        [f"{m}/{d}/{y}", f"{m:02d}/{d:02d}/{y}", f"{abbr} {d} {y}"],
        density,
    )


def _render_leg(rng: random.Random, line: ShipmentLine, knobs: Knobs) -> str:
    """Render one shipment line into a freetext phrase (no reference)."""
    density = knobs.abbrev_density
    parts: list[str] = [
        _leg_word(rng, line.leg, density),
        f"{line.quantity} {_unit_word(rng, line.unit, density)}",
    ]
    lane = _lane_phrase(rng, line.origin, line.destination, density)
    if lane:
        parts.append(lane)
    if line.weight is not None:
        parts.append(_weight_phrase(rng, line.weight, density))
    date = _date_phrase(rng, line, density)
    if date:
        parts.append(date)
    parts.append(_rate_phrase(rng, line.rate, density))
    if line.accessorial is not None:
        parts.append(_accessorial_phrase(rng, line.accessorial, density))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Style renderers. Each takes the already-built lines and produces the final
# text. The reference is rendered VERBATIM so it is always a literal substring.
# ---------------------------------------------------------------------------


def _render_terse(rng: random.Random, lines: list[ShipmentLine], knobs: Knobs, ref: str | None) -> str:
    text = _render_leg(rng, lines[0], knobs)
    if ref:
        text = f"{text} {ref}"
    return text


def _render_verbose(rng: random.Random, lines: list[ShipmentLine], knobs: Knobs, ref: str | None) -> str:
    carrier = rng.choice(CARRIERS)
    commodity = rng.choice(COMMODITIES)
    text = f"{carrier} shipment confirmation. {_render_leg(rng, lines[0], knobs)}. Commodity: {commodity}."
    if ref:
        text += f" Reference {ref}."
    return text


def _render_abbrev(rng: random.Random, lines: list[ShipmentLine], knobs: Knobs, ref: str | None) -> str:
    text = _render_leg(rng, lines[0], knobs)
    if rng.random() < 0.5:
        text = f"{rng.choice(CARRIERS)} - {text}"
    if ref:
        text += f" REF {ref}"
    return text


def _render_multileg(rng: random.Random, lines: list[ShipmentLine], knobs: Knobs, ref: str | None) -> str:
    """A multi-stop confirmation: header + one block per leg.

    The reference is stated in the header *and* repeated on every leg — mirroring
    how a real multi-stop confirmation restates the load/BOL number at each stop.
    """
    carrier = rng.choice(CARRIERS)
    header = f"{carrier} multi-stop load confirmation"
    if ref:
        header += f" (ref {ref})"
    blocks: list[str] = []
    for i, line in enumerate(lines, start=1):
        block = f"Leg {i}: {_render_leg(rng, line, knobs)}"
        if ref:
            block += f" [ref {ref}]"
        blocks.append(block)
    return header + "\n" + "\n".join(blocks)


_RENDERERS = {
    "terse": _render_terse,
    "verbose": _render_verbose,
    "abbrev": _render_abbrev,
    "multileg": _render_multileg,
}


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def build_example(
    rng: random.Random,
    index: int,
    style: str,
    complexity: int,
    *,
    seed: int = 0,
) -> dict:
    """Build one example dict: label-first, then render.

    Returns ``{"id", "text", "parsed_json", "style", "complexity"}`` where
    ``parsed_json`` is the canonical JSON produced by
    :func:`freight_schema.dump_lines`. The label is correct by construction, and
    any non-null reference is guaranteed to be a literal substring of ``text``.
    """
    if style not in _RENDERERS:
        raise ValueError(f"unknown style: {style!r}")

    knobs = knobs_for(style, complexity, rng)

    # One reference is stated once and applies to every line (see ShipmentLine).
    # Only about half of real confirmations carry an explicit PO/BOL number.
    reference = _make_reference(rng) if rng.random() < 0.45 else None

    lines: list[ShipmentLine] = []
    for leg_index in range(knobs.n_legs):
        if style == "multileg":
            leg = "pickup" if leg_index == 0 else "delivery"
        else:
            leg = rng.choice(("pickup", "delivery"))
        lines.append(_build_line(rng, leg, style, knobs, reference))

    text = _RENDERERS[style](rng, lines, knobs, reference)

    # Correct-by-construction guard: the reference we stored must appear in the
    # text. Because we render it verbatim this always holds; the assert protects
    # future edits to the renderers.
    assert reference is None or reference in text, "reference must be a literal substring of text"

    return {
        "id": f"synth-{seed}-{index:05d}",
        "text": text,
        "parsed_json": dump_lines(lines),
        "style": style,
        "complexity": complexity,
    }


def generate_examples(n: int, seed: int) -> list[dict]:
    """Generate ``n`` synthetic examples deterministically from ``seed``.

    A single ``random.Random(seed)`` drives every choice, so identical
    ``(n, seed)`` inputs always yield identical output. Each element is the dict
    returned by :func:`build_example`.
    """
    rng = random.Random(seed)
    examples: list[dict] = []
    for i in range(n):
        style = rng.choice(STYLES)
        complexity = rng.randint(1, MAX_COMPLEXITY)
        examples.append(build_example(rng, i, style, complexity, seed=seed))
    return examples
