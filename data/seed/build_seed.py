"""Author the hand-written seed dataset for the freight-parser demo.

This is the *root* of the whole data pipeline. Everything downstream -- the QA
portal's initial rows, the synthetic generator's few-shot examples, the trainer's
alpaca dataset, the evaluator's benchmark -- ultimately traces back to these
~60 hand-written shipment confirmations and their gold parses.

Why author gold as real :class:`ShipmentLine` objects instead of typing JSON by
hand?  Because then the labels *cannot* be invalid: every gold record is
constructed through the shared pydantic schema and serialized with the shared
``dump_lines`` helper, so the JSON on disk is byte-for-byte the same shape the
model is trained to emit and the evaluator compares against. Hand-typed JSON
drifts; schema-built JSON can't.

Run it to (re)generate ``data/seed/confirmations.jsonl``::

    uv run python data/seed/build_seed.py

Output format (one JSON object per line -- the "Seed" format from the contract)::

    {"id": "seed-001", "text": "<raw messy confirmation>", "gold": [<ShipmentLine dict>, ...]}

The domain is freight/logistics only. Carriers, references, lanes, and
commodities below are all invented, clean-room, synthetic -- no real companies.

--------------------------------------------------------------------------------
Field conventions honored by every example (see freight_schema.models):
  * Partial dates keep ONLY the parts the text states. "Apr 5th" -> month+day
    (year stays null); "MAR" -> month only; "3/14/24" -> full m/d/y; no date at
    all -> all three null. We never fabricate a part the text does not give.
    Two-digit years are expanded to 20xx ("24" -> 2024) -- a small, teachable
    normalization the model can learn.
  * All date fields live under ``pickup_*`` (the schema's only date slot), so we
    only attach a date to the leg that is actually picked up. Delivery legs carry
    no date here.
  * Abbreviations normalize on the ``leg`` field: PU/pu/pickup -> "pickup";
    DLV/dlv/del/deliver/delivery -> "delivery". The raw text deliberately uses a
    mix so the model sees every variant.
  * ``rate`` is required and stored exactly as the negotiated number stated
    (linehaul like 1,847.50, per-mile like 2.16, per-cwt like 24.18, etc.). We
    keep the numeric value; the "/mi" or "cwt" unit is context, not part of rate.
  * ``weight`` is the weight figure as stated ("4,200 lb" -> 4200; "@940lbs ea"
    -> 940). Absent -> null.
  * ``accessorial`` (fuel surcharge / detention / lumper) is a dollar amount when
    stated, else null -- present on a minority of messages.
  * ``reference`` (PO/BOL/acct) is stated ONCE per message and repeated on every
    line of that message. It MUST appear literally in the text (the serving layer
    nulls any reference that is not a verbatim substring -- see
    ``apply_reference_guard``), so we only ever set it to a real substring here.
"""

from __future__ import annotations

import json
from decimal import Decimal

from freight_schema import ShipmentLine, dump_lines
from freight_schema.paths import seed_file

# Short alias: money/weight values are authored as exact Decimals (from strings,
# never floats, so "1847.50" stays 1847.50 and never becomes 1847.4999...).
D = Decimal


def line(
    quantity: int,
    unit: str,
    leg: str,
    rate: Decimal,
    *,
    origin: str | None = None,
    destination: str | None = None,
    weight: Decimal | None = None,
    pickup_month: int | None = None,
    pickup_day: int | None = None,
    pickup_year: int | None = None,
    accessorial: Decimal | None = None,
    reference: str | None = None,
) -> ShipmentLine:
    """Build one gold :class:`ShipmentLine`.

    Keyword-only optionals keep each confirmation below readable: you only spell
    out the fields the text actually contains, and everything absent defaults to
    ``None`` -- which is exactly the label we want for an unstated field.
    """
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


# =============================================================================
# The confirmations.
#
# Each entry is (raw_text, [gold ShipmentLine, ...]). The raw text is messy and
# varied on purpose -- terse vs verbose, abbreviation-heavy vs spelled-out, lane
# codes (CHI/LAX/DAL/MEM/ATL/DFW) vs full city names, single-leg vs multi-leg.
# The gold list is the correct parse. IDs are assigned automatically below, so
# the list stays easy to reorder and extend.
# =============================================================================

CONFIRMATIONS: list[tuple[str, list[ShipmentLine]]] = [
    # --- 001: baseline terse single-leg pickup, comma rate, no date/ref/weight.
    (
        "ACME FREIGHT - PU 12 pallets CHI -> LAX, rate 1,847.50",
        [line(12, "pallets", "pickup", D("1847.50"), origin="CHI", destination="LAX")],
    ),
    # --- 002: single-leg delivery, spelled-out cities, "delivery" abbreviation.
    (
        "ORION LOGISTICS delivery 6 skids from Columbus to Indianapolis. Rate 985.00",
        [line(6, "skids", "delivery", D("985.00"), origin="Columbus", destination="Indianapolis")],
    ),
    # --- 003: PARTIAL month+day ("Apr 5th" -> 4/5, year null) + weight "4,200 lb".
    (
        "MERIDIAN CARRIERS: PU Apr 5th, 20 cartons DAL to MEM, weight 4,200 lb, rate 2,150.00",
        [
            line(20, "cartons", "pickup", D("2150.00"), origin="DAL", destination="MEM",
                 weight=D("4200"), pickup_month=4, pickup_day=5),
        ],
    ),
    # --- 004: PARTIAL month only ("MAR" -> month=3, day/year null).
    (
        "SUMMIT HAULING - ship in MAR, 3 containers LAX -> Denver, rate 3,250.00",
        [line(3, "containers", "pickup", D("3250.00"), origin="LAX", destination="Denver",
              pickup_month=3)],
    ),
    # --- 005: FULL date "3/14/24" -> 3/14/2024 (two-digit year expanded).
    (
        "BLUE RIDGE TRANSPORT PU 3/14/24 - 10 pallets ATL to Charlotte @ rate 1,425.00",
        [line(10, "pallets", "pickup", D("1425.00"), origin="ATL", destination="Charlotte",
              pickup_month=3, pickup_day=14, pickup_year=2024)],
    ),
    # --- 006: reference stated once (PO-88231), single-leg delivery.
    (
        "CASCADE FREIGHTWAYS. Ref PO-88231. Deliver 8 skids Seattle to Portland. Rate 640.00",
        [line(8, "skids", "delivery", D("640.00"), origin="Seattle", destination="Portland",
              reference="PO-88231")],
    ),
    # --- 007: MULTI-LEG (2), shared reference BOL-4471 repeated on both lines.
    (
        "IRONWOOD LOGISTICS - BOL-4471\n"
        "Leg 1: PU 15 pallets CHI -> DFW, rate 1,900.00\n"
        "Leg 2: DLV 15 pallets DFW -> Kansas City, rate 1,250.00",
        [
            line(15, "pallets", "pickup", D("1900.00"), origin="CHI", destination="DFW",
                 reference="BOL-4471"),
            line(15, "pallets", "delivery", D("1250.00"), origin="DFW", destination="Kansas City",
                 reference="BOL-4471"),
        ],
    ),
    # --- 008: ACCESSORIAL (fuel surcharge 125.50), single-leg pickup.
    (
        "NORTHSTAR CARTAGE: pu 5 containers Newark to ATL, fuel surcharge 125.50, rate 2,960.75",
        [line(5, "containers", "pickup", D("2960.75"), origin="Newark", destination="ATL",
              accessorial=D("125.50"))],
    ),
    # --- 009: terse single-leg delivery, "del" abbreviation, no extras.
    (
        "PIONEER TRANSIT del 40 cartons MEM to ATL rate 1,120.00",
        [line(40, "cartons", "delivery", D("1120.00"), origin="MEM", destination="ATL")],
    ),
    # --- 010: tricky per-mile rate 2.16, heavy weight "18000 lb".
    (
        "VANGUARD FREIGHT LINES - PU 2 containers Phoenix to DAL, 18000 lb, rate 2.16/mi",
        [line(2, "containers", "pickup", D("2.16"), origin="Phoenix", destination="DAL",
              weight=D("18000"))],
    ),
    # --- 011: reference PO#55120 + per-each weight "@940lbs ea".
    (
        "KESTREL LOGISTICS PO#55120 - pickup 24 skids @940lbs ea, CHI to Cleveland, rate 1,660.00",
        [line(24, "skids", "pickup", D("1660.00"), origin="CHI", destination="Cleveland",
              weight=D("940"), reference="PO#55120")],
    ),
    # --- 012: PARTIAL month+day ("Jun 12"), single-leg pickup.
    (
        "GRANITE STATE CARRIERS - PU Jun 12: 9 pallets Denver -> Kansas City, rate 1,540.00",
        [line(9, "pallets", "pickup", D("1540.00"), origin="Denver", destination="Kansas City",
              pickup_month=6, pickup_day=12)],
    ),
    # --- 013: reference "acct 30291" (account-style ref), delivery.
    (
        "HARBOR POINT FREIGHT (acct 30291) - deliver 14 cartons Portland to Seattle, rate 720.00",
        [line(14, "cartons", "delivery", D("720.00"), origin="Portland", destination="Seattle",
              reference="acct 30291")],
    ),
    # --- 014: PARTIAL month only ("in AUG" -> month=8).
    (
        "REDWOOD DISPATCH - moving in AUG, 30 pallets DFW to Denver, rate 2,480.00",
        [line(30, "pallets", "pickup", D("2480.00"), origin="DFW", destination="Denver",
              pickup_month=8)],
    ),
    # --- 015: MULTI-LEG (3), shared reference PO-2025-0417 on all three lines.
    (
        "ATLAS OVERLAND - PO-2025-0417\n"
        "1) PU 10 containers LAX -> Phoenix, rate 1,150.00\n"
        "2) DLV 10 containers Phoenix -> DAL, rate 1,340.00\n"
        "3) DLV 10 containers DAL -> MEM, rate 980.00",
        [
            line(10, "containers", "pickup", D("1150.00"), origin="LAX", destination="Phoenix",
                 reference="PO-2025-0417"),
            line(10, "containers", "delivery", D("1340.00"), origin="Phoenix", destination="DAL",
                 reference="PO-2025-0417"),
            line(10, "containers", "delivery", D("980.00"), origin="DAL", destination="MEM",
                 reference="PO-2025-0417"),
        ],
    ),
    # --- 016: tricky per-cwt rate 24.18, weight "12,000 lbs", terse.
    (
        "CEDAR LINE HAULERS pu 18 skids ATL->CHI 12,000 lbs rate 24.18 cwt",
        [line(18, "skids", "pickup", D("24.18"), origin="ATL", destination="CHI",
              weight=D("12000"))],
    ),
    # --- 017: verbose, reference "BOL 77-2231" (has a space), detention accessorial.
    (
        "ACME FREIGHT confirmation. Reference BOL 77-2231. Please pick up 7 pallets at our "
        "Cincinnati dock bound for Cleveland. Detention 75.00 if held. Line rate 1,010.00.",
        [line(7, "pallets", "pickup", D("1010.00"), origin="Cincinnati", destination="Cleveland",
              accessorial=D("75.00"), reference="BOL 77-2231")],
    ),
    # --- 018: FULL date with 4-digit year "5/2/2026".
    (
        "ORION LOGISTICS - PU 5/2/2026, 22 cartons CHI to DAL, rate 1,775.00",
        [line(22, "cartons", "pickup", D("1775.00"), origin="CHI", destination="DAL",
              pickup_month=5, pickup_day=2, pickup_year=2026)],
    ),
    # --- 019: NO lane stated -> origin/destination null (both optional).
    (
        "MERIDIAN CARRIERS - dlv 16 pallets, rate 1,300.00 (lane TBD)",
        [line(16, "pallets", "delivery", D("1300.00"))],
    ),
    # --- 020: MULTI-LEG (2), no reference, mixed pu/del abbreviations.
    (
        "SUMMIT HAULING\n"
        "pu 12 skids Kansas City -> DAL rate 1,220.00\n"
        "del 12 skids DAL -> ATL rate 1,410.00",
        [
            line(12, "skids", "pickup", D("1220.00"), origin="Kansas City", destination="DAL"),
            line(12, "skids", "delivery", D("1410.00"), origin="DAL", destination="ATL"),
        ],
    ),
    # --- 021: PARTIAL month+day ("Sep 9th") + reference PO-91002.
    (
        "BLUE RIDGE TRANSPORT - PO-91002 - PU Sep 9th 11 containers Seattle to Denver, rate 3,090.00",
        [line(11, "containers", "pickup", D("3090.00"), origin="Seattle", destination="Denver",
              pickup_month=9, pickup_day=9, reference="PO-91002")],
    ),
    # --- 022: tricky low rate 0.50 (per-lb), weight, no ref/date.
    (
        "CASCADE FREIGHTWAYS pu 4 skids steel coils DAL to MEM, 8,000 lbs, rate 0.50/lb",
        [line(4, "skids", "pickup", D("0.50"), origin="DAL", destination="MEM",
              weight=D("8000"))],
    ),
    # --- 023: verbose delivery, reference BOL-5588, commodity string.
    (
        "IRONWOOD LOGISTICS - kindly confirm delivery of 28 cartons of packaged goods from "
        "Indianapolis to Columbus. BOL-5588. Rate agreed at 1,150.00.",
        [line(28, "cartons", "delivery", D("1150.00"), origin="Indianapolis", destination="Columbus",
              reference="BOL-5588")],
    ),
    # --- 024: PARTIAL month only on first line ("NOV") + MULTI-LEG (2) + ref acct#GT-4410.
    (
        "NORTHSTAR CARTAGE acct#GT-4410\n"
        "PU NOV - 20 pallets CHI -> DFW rate 1,880.00\n"
        "DLV - 20 pallets DFW -> LAX rate 2,540.00",
        [
            line(20, "pallets", "pickup", D("1880.00"), origin="CHI", destination="DFW",
                 pickup_month=11, reference="acct#GT-4410"),
            line(20, "pallets", "delivery", D("2540.00"), origin="DFW", destination="LAX",
                 reference="acct#GT-4410"),
        ],
    ),
    # --- 025: detention accessorial 150.00, single-leg delivery.
    (
        "PIONEER TRANSIT - del 9 containers Portland to Phoenix, detention 150.00, rate 3,410.00",
        [line(9, "containers", "delivery", D("3410.00"), origin="Portland", destination="Phoenix",
              accessorial=D("150.00"))],
    ),
    # --- 026: abbreviation-heavy terse ("plts", ">", "rt"), no extras.
    (
        "VANGUARD FL - pu 6 plts CHI>ATL rt 1,090.00",
        [line(6, "pallets", "pickup", D("1090.00"), origin="CHI", destination="ATL")],
    ),
    # --- 027: "dlv" abbreviation + per-each weight "@ 1,050 lbs each".
    (
        "KESTREL LOGISTICS dlv 12 skids Denver to DAL @ 1,050 lbs each, rate 1,470.00",
        [line(12, "skids", "delivery", D("1470.00"), origin="Denver", destination="DAL",
              weight=D("1050"))],
    ),
    # --- 028: PARTIAL month+day on first line ("Oct 3rd") + MULTI-LEG (3), no ref.
    (
        "GRANITE STATE CARRIERS\n"
        "Leg A: PU Oct 3rd 8 pallets ATL -> MEM rate 980.00\n"
        "Leg B: DLV 8 pallets MEM -> DAL rate 1,140.00\n"
        "Leg C: DLV 8 pallets DAL -> Phoenix rate 1,620.00",
        [
            line(8, "pallets", "pickup", D("980.00"), origin="ATL", destination="MEM",
                 pickup_month=10, pickup_day=3),
            line(8, "pallets", "delivery", D("1140.00"), origin="MEM", destination="DAL"),
            line(8, "pallets", "delivery", D("1620.00"), origin="DAL", destination="Phoenix"),
        ],
    ),
    # --- 029: FULL date "12/9/25" + reference PO-30044.
    (
        "HARBOR POINT FREIGHT PO-30044 - PU 12/9/25, 5 containers LAX to Seattle, rate 1,980.00",
        [line(5, "containers", "pickup", D("1980.00"), origin="LAX", destination="Seattle",
              pickup_month=12, pickup_day=9, pickup_year=2025, reference="PO-30044")],
    ),
    # --- 030: weight "gross 2,750 lbs", no date/ref.
    (
        "REDWOOD DISPATCH pu 15 cartons Cincinnati to Cleveland, gross 2,750 lbs, rate 690.00",
        [line(15, "cartons", "pickup", D("690.00"), origin="Cincinnati", destination="Cleveland",
              weight=D("2750"))],
    ),
    # --- 031: MULTI-LEG (2) with shared accessorial 210.00 AND shared reference BOL-6620.
    (
        "ATLAS OVERLAND - BOL-6620 - fuel surcharge 210.00 applies both legs\n"
        "PU 30 pallets CHI -> DAL rate 1,900.00\n"
        "DLV 30 pallets DAL -> LAX rate 2,650.00",
        [
            line(30, "pallets", "pickup", D("1900.00"), origin="CHI", destination="DAL",
                 accessorial=D("210.00"), reference="BOL-6620"),
            line(30, "pallets", "delivery", D("2650.00"), origin="DAL", destination="LAX",
                 accessorial=D("210.00"), reference="BOL-6620"),
        ],
    ),
    # --- 032: per-cwt rate 19.75, weight "15,400 lbs", delivery.
    (
        "CEDAR LINE HAULERS - dlv 22 skids MEM to ATL, 15,400 lbs, rate 19.75 cwt",
        [line(22, "skids", "delivery", D("19.75"), origin="MEM", destination="ATL",
              weight=D("15400"))],
    ),
    # --- 033: PARTIAL month only ("JAN") + reference "acct 55210".
    (
        "ACME FREIGHT acct 55210 - PU JAN, 18 cartons DFW to CHI, rate 1,760.00",
        [line(18, "cartons", "pickup", D("1760.00"), origin="DFW", destination="CHI",
              pickup_month=1, reference="acct 55210")],
    ),
    # --- 034: NO lane (origin/destination null), delivery.
    (
        "ORION LOGISTICS - DLV 25 pallets (destination on BOL), rate 1,505.00",
        [line(25, "pallets", "delivery", D("1505.00"))],
    ),
    # --- 035: MULTI-LEG (2) with a partial date on EACH leg + shared ref PO-77881.
    (
        "MERIDIAN CARRIERS PO-77881\n"
        "PU Apr 5th 14 containers Seattle -> LAX rate 2,100.00\n"
        "DLV Apr 8th 14 containers LAX -> Phoenix rate 1,350.00",
        [
            line(14, "containers", "pickup", D("2100.00"), origin="Seattle", destination="LAX",
                 pickup_month=4, pickup_day=5, reference="PO-77881"),
            line(14, "containers", "delivery", D("1350.00"), origin="LAX", destination="Phoenix",
                 pickup_month=4, pickup_day=8, reference="PO-77881"),
        ],
    ),
    # --- 036: terse single-leg pickup, containers, no extras.
    (
        "SUMMIT HAULING pu 3 containers ATL->DAL rate 1,240.00",
        [line(3, "containers", "pickup", D("1240.00"), origin="ATL", destination="DAL")],
    ),
    # --- 037: PARTIAL month+day ("May 20th") + weight "16,800 lb".
    (
        "BLUE RIDGE TRANSPORT - PU May 20th, 40 pallets CHI to MEM, 16,800 lb, rate 2,300.00",
        [line(40, "pallets", "pickup", D("2300.00"), origin="CHI", destination="MEM",
              weight=D("16800"), pickup_month=5, pickup_day=20)],
    ),
    # --- 038: verbose, reference BOL-9001, detention accessorial, commodity string.
    (
        "CASCADE FREIGHTWAYS. Our reference is BOL-9001. Deliver 11 skids of industrial adhesive "
        "from Denver to Kansas City. Detention billed at 90.00/hr after 2 hours free. "
        "Agreed rate 1,180.00.",
        [line(11, "skids", "delivery", D("1180.00"), origin="Denver", destination="Kansas City",
              accessorial=D("90.00"), reference="BOL-9001")],
    ),
    # --- 039: MULTI-LEG (2), no ref, mixed PU/dlv abbreviations, ">" lanes.
    (
        "IRONWOOD LOGISTICS\n"
        "PU 9 cartons Columbus>Cincinnati rate 540.00\n"
        "dlv 9 cartons Cincinnati>Cleveland rate 610.00",
        [
            line(9, "cartons", "pickup", D("540.00"), origin="Columbus", destination="Cincinnati"),
            line(9, "cartons", "delivery", D("610.00"), origin="Cincinnati", destination="Cleveland"),
        ],
    ),
    # --- 040: reference PO-2024-1188, single-leg pickup, no date.
    (
        "NORTHSTAR CARTAGE PO-2024-1188 - pu 50 cartons LAX to DFW, rate 2,050.00",
        [line(50, "cartons", "pickup", D("2050.00"), origin="LAX", destination="DFW",
              reference="PO-2024-1188")],
    ),
    # --- 041: PARTIAL month only ("in SEP"), single-leg pickup.
    (
        "PIONEER TRANSIT - PU in SEP, 7 pallets Phoenix to LAX, rate 1,015.00",
        [line(7, "pallets", "pickup", D("1015.00"), origin="Phoenix", destination="LAX",
              pickup_month=9)],
    ),
    # --- 042: MULTI-LEG (3), no ref, no dates -- a long relay.
    (
        "VANGUARD FREIGHT LINES\n"
        "1) PU 20 pallets CHI -> DAL rate 1,850.00\n"
        "2) DLV 20 pallets DAL -> ATL rate 1,410.00\n"
        "3) DLV 20 pallets ATL -> MEM rate 990.00",
        [
            line(20, "pallets", "pickup", D("1850.00"), origin="CHI", destination="DAL"),
            line(20, "pallets", "delivery", D("1410.00"), origin="DAL", destination="ATL"),
            line(20, "pallets", "delivery", D("990.00"), origin="ATL", destination="MEM"),
        ],
    ),
    # --- 043: the "everything" line -- weight, fuel surcharge, and reference PO-44120.
    (
        "KESTREL LOGISTICS - Ref PO-44120 - PU 6 containers Newark to CHI, 40,000 lbs, "
        "fuel surcharge 305.25, rate 3,880.00",
        [line(6, "containers", "pickup", D("3880.00"), origin="Newark", destination="CHI",
              weight=D("40000"), accessorial=D("305.25"), reference="PO-44120")],
    ),
    # --- 044: terse single-leg delivery, "del" abbreviation, no extras.
    (
        "GRANITE STATE CARRIERS del 30 skids DFW->CHI rate 1,960.00",
        [line(30, "skids", "delivery", D("1960.00"), origin="DFW", destination="CHI")],
    ),
    # --- 045: PARTIAL month+day on first line ("Feb 14th") + MULTI-LEG (2) + ref BOL-3312.
    (
        "HARBOR POINT FREIGHT - BOL-3312\n"
        "PU Feb 14th 12 pallets ATL -> Charlotte rate 860.00\n"
        "DLV 12 pallets Charlotte -> Columbus rate 1,050.00",
        [
            line(12, "pallets", "pickup", D("860.00"), origin="ATL", destination="Charlotte",
                 pickup_month=2, pickup_day=14, reference="BOL-3312"),
            line(12, "pallets", "delivery", D("1050.00"), origin="Charlotte", destination="Columbus",
                 reference="BOL-3312"),
        ],
    ),
    # --- 046: weight "36,000 lb", no date/ref, comma rate.
    (
        "REDWOOD DISPATCH pu 8 containers LAX to Denver, 36,000 lb, rate 2,725.00",
        [line(8, "containers", "pickup", D("2725.00"), origin="LAX", destination="Denver",
              weight=D("36000"))],
    ),
    # --- 047: verbose delivery, flat detention accessorial 60.00, commodity string.
    (
        "ATLAS OVERLAND - Please deliver 19 cartons of packaged snacks into Cleveland from "
        "Cincinnati. Detention 60.00 flat. Rate 780.00.",
        [line(19, "cartons", "delivery", D("780.00"), origin="Cincinnati", destination="Cleveland",
              accessorial=D("60.00"))],
    ),
    # --- 048: per-mile rate 1.95 + reference "acct 88120".
    (
        "CEDAR LINE HAULERS acct 88120 - pu 2 containers Seattle to Phoenix, rate 1.95/mi",
        [line(2, "containers", "pickup", D("1.95"), origin="Seattle", destination="Phoenix",
              reference="acct 88120")],
    ),
    # --- 049: NO lane, no date, no ref -- minimal delivery line.
    (
        "ORION LOGISTICS - dlv 33 cartons, rate 1,240.00",
        [line(33, "cartons", "delivery", D("1240.00"))],
    ),
    # --- 050: FULL date "7/2/26" + weight "9,800 lbs" + reference PO-50021.
    (
        "ACME FREIGHT PO-50021 - PU 7/2/26, 14 pallets CHI to LAX, 9,800 lbs, rate 2,010.00",
        [line(14, "pallets", "pickup", D("2010.00"), origin="CHI", destination="LAX",
              weight=D("9800"), pickup_month=7, pickup_day=2, pickup_year=2026,
              reference="PO-50021")],
    ),
    # --- 051: MULTI-LEG (2), shared reference BOL-7742, mixed pu/del.
    (
        "MERIDIAN CARRIERS - BOL-7742\n"
        "pu 25 skids MEM -> ATL rate 1,330.00\n"
        "del 25 skids ATL -> DAL rate 1,220.00",
        [
            line(25, "skids", "pickup", D("1330.00"), origin="MEM", destination="ATL",
                 reference="BOL-7742"),
            line(25, "skids", "delivery", D("1220.00"), origin="ATL", destination="DAL",
                 reference="BOL-7742"),
        ],
    ),
    # --- 052: detention accessorial 120.00, single-leg pickup.
    (
        "SUMMIT HAULING - pu 4 containers DAL to Phoenix, detention 120.00, rate 3,150.00",
        [line(4, "containers", "pickup", D("3150.00"), origin="DAL", destination="Phoenix",
              accessorial=D("120.00"))],
    ),
    # --- 053: terse single-leg delivery, spelled-out cities, no extras.
    (
        "BLUE RIDGE TRANSPORT del 17 pallets Kansas City to Denver rate 1,090.00",
        [line(17, "pallets", "delivery", D("1090.00"), origin="Kansas City", destination="Denver")],
    ),
    # --- 054: MULTI-LEG (3), shared reference PO-2026-0055, weight on the first leg.
    (
        "CASCADE FREIGHTWAYS - PO-2026-0055\n"
        "1) PU 10 pallets CHI -> DFW, 12,000 lb, rate 1,700.00\n"
        "2) DLV 10 pallets DFW -> LAX, rate 2,400.00\n"
        "3) DLV 10 pallets LAX -> Seattle, rate 1,560.00",
        [
            line(10, "pallets", "pickup", D("1700.00"), origin="CHI", destination="DFW",
                 weight=D("12000"), reference="PO-2026-0055"),
            line(10, "pallets", "delivery", D("2400.00"), origin="DFW", destination="LAX",
                 reference="PO-2026-0055"),
            line(10, "pallets", "delivery", D("1560.00"), origin="LAX", destination="Seattle",
                 reference="PO-2026-0055"),
        ],
    ),
    # --- 055: abbreviation-heavy terse ("ctns", ">", "rt"), delivery.
    (
        "IRONWOOD LOG - dlv 21 ctns Columbus>Indianapolis rt 640.00",
        [line(21, "cartons", "delivery", D("640.00"), origin="Columbus", destination="Indianapolis")],
    ),
    # --- 056: per-each weight "@ 875 lbs ea" + reference BOL-1200.
    (
        "NORTHSTAR CARTAGE BOL-1200 - pu 16 skids @ 875 lbs ea, ATL to MEM, rate 1,410.00",
        [line(16, "skids", "pickup", D("1410.00"), origin="ATL", destination="MEM",
              weight=D("875"), reference="BOL-1200")],
    ),
    # --- 057: fuel surcharge accessorial 175.00, no ref/date, single-leg pickup.
    (
        "PIONEER TRANSIT - PU 7 containers Phoenix to DFW, fuel surcharge 175.00, rate 3,020.00",
        [line(7, "containers", "pickup", D("3020.00"), origin="Phoenix", destination="DFW",
              accessorial=D("175.00"))],
    ),
    # --- 058: verbose finale -- MULTI-LEG (2) with partial date, weight, accessorial, and
    #          reference PO-99010 all on the first leg; commodity string; spelled-out legs.
    (
        "VANGUARD FREIGHT LINES - confirmation PO-99010.\n"
        "Leg 1: pickup on Mar 30th, 12 pallets of ceramic tile, Denver to DAL, 22,500 lbs, "
        "rate 2,240.00.\n"
        "Leg 2: delivery, 12 pallets, DAL to ATL, rate 1,480.00.\n"
        "Fuel surcharge 140.00 on leg 1.",
        [
            line(12, "pallets", "pickup", D("2240.00"), origin="Denver", destination="DAL",
                 weight=D("22500"), pickup_month=3, pickup_day=30, accessorial=D("140.00"),
                 reference="PO-99010"),
            line(12, "pallets", "delivery", D("1480.00"), origin="DAL", destination="ATL",
                 reference="PO-99010"),
        ],
    ),
]


def build() -> str:
    """Serialize every confirmation to the seed JSONL file and return its path.

    Each output line is ``{"id", "text", "gold"}``. The gold list is produced by
    the shared ``dump_lines`` (Decimals as JSON numbers, canonical field order)
    and parsed back with ``json.loads`` so it embeds as a real JSON array -- not a
    quoted string -- inside the record.
    """
    out_path = seed_file()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[str] = []
    for i, (text, lines) in enumerate(CONFIRMATIONS, start=1):
        gold = json.loads(dump_lines(lines))  # canonical, schema-validated labels
        record = {"id": f"seed-{i:03d}", "text": text, "gold": gold}
        records.append(json.dumps(record, ensure_ascii=True))

    out_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    return str(out_path)


if __name__ == "__main__":
    path = build()
    print(f"Wrote {len(CONFIRMATIONS)} confirmations to {path}")
