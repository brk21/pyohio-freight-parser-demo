# `data/` — the freight-parser data pipeline

Everything here is **synthetic, clean-room freight/logistics data**. Carriers
(ACME FREIGHT, ORION LOGISTICS, MERIDIAN CARRIERS, …), reference numbers, lanes,
and commodities are all invented for the demo. No real companies, no real
shipments, no real people.

The pipeline flows in one direction, and it all starts from the hand-written
seed file:

```
data/seed/confirmations.jsonl   <-- authored here (build_seed.py)
        |
        v   (loaded into the QA portal, few-shot for the synthetic generator)
data/synthetic.duckdb           <-- machine-generated examples
        |
        v   (prep_dataset merges seed + synthetic)
data/training/train.jsonl       <-- alpaca-format fine-tuning set
data/training/benchmark.jsonl   <-- held-out eval set
        |
        v   (train -> predict -> score)
data/adapters/<model>/          <-- trained LoRA adapters
data/predictions/<model>.jsonl  <-- scored benchmark predictions
```

This README documents the **seed** stage (`data/seed/`). The other stages own
their own formats (see the build contract), but every one of them ultimately
speaks the same `freight_schema.ShipmentLine` shape.

---

## The seed format

`data/seed/confirmations.jsonl` is [JSON Lines](https://jsonlines.org/): one JSON
object per line. Each object is a single carrier confirmation plus its correct
("gold") parse:

```json
{
  "id": "seed-007",
  "text": "IRONWOOD LOGISTICS - BOL-4471\nLeg 1: PU 15 pallets CHI -> DFW, rate 1,900.00\nLeg 2: DLV 15 pallets DFW -> Kansas City, rate 1,250.00",
  "gold": [
    {"origin": "CHI", "destination": "DFW", "quantity": 15, "unit": "pallets", "weight": null, "pickup_day": null, "pickup_month": null, "pickup_year": null, "leg": "pickup", "accessorial": null, "rate": 1900.0, "reference": "BOL-4471"},
    {"origin": "DFW", "destination": "Kansas City", "quantity": 15, "unit": "pallets", "weight": null, "pickup_day": null, "pickup_month": null, "pickup_year": null, "leg": "delivery", "accessorial": null, "rate": 1250.0, "reference": "BOL-4471"}
  ]
}
```

| field   | meaning                                                                             |
| ------- | ----------------------------------------------------------------------------------- |
| `id`    | Stable identifier, `seed-NNN`.                                                       |
| `text`  | The raw, messy confirmation exactly as a freight partner might send it.             |
| `gold`  | A **list** of `ShipmentLine` dicts — one per leg. A confirmation may be multi-leg.  |

Every `gold` entry is a full `ShipmentLine` dict (see
`packages/schema/freight_schema/models.py`); unstated fields are `null`.

### How it's built (and why)

`data/seed/confirmations.jsonl` is **generated** — do not hand-edit it. It is
authored in `data/seed/build_seed.py`, where each confirmation is written as a
`(raw_text, [ShipmentLine, ...])` pair using the real `freight_schema` models,
then serialized with `freight_schema.dump_lines`. Building the labels out of live
schema objects means they *cannot* be structurally invalid — the JSON on disk is
exactly the shape the model is trained to emit and the evaluator compares against.

Regenerate it any time with:

```bash
uv run python data/seed/build_seed.py
```

### Field conventions the seed deliberately exercises

The seed is small but chosen to cover every part of the problem that is easy to
get wrong:

- **Multi-leg vs single-leg.** Some confirmations are one line; others describe a
  2–3 leg relay, so the top-level parse is always a list.
- **Partial dates — keep only what's stated, never fabricate.**
  `"Apr 5th"` → `pickup_month=4, pickup_day=5` (year stays `null`); `"MAR"` →
  `pickup_month=3` only; `"3/14/24"` → full `3 / 14 / 2024` (two-digit years
  expand to `20xx`); no date → all three `null`. Dates live only on the leg that
  is picked up (the schema's single `pickup_*` date slot).
- **Abbreviations normalize on `leg`.** `PU` / `pu` / `pickup` → `"pickup"`;
  `DLV` / `dlv` / `del` / `deliver` / `delivery` → `"delivery"`. The raw text
  uses a mix.
- **Optional fields go `null` when absent** — weight, accessorial, reference, and
  dates are all frequently unstated.
- **Accessorials** (fuel surcharge, detention, lumper) appear as a dollar amount
  on a minority of messages, `null` otherwise.
- **References stated once, applied to all legs.** A `PO-…` / `BOL-…` / `acct …`
  number is written once in the message and repeated on every `gold` line of that
  confirmation. **A non-null `reference` MUST appear verbatim in `text`** — the
  serving layer's `apply_reference_guard` nulls any reference that is not a
  literal substring of the input, so authored gold never invents one. Roughly
  half the confirmations carry a reference.
- **Exact numbers.** Rates and weights are matched exactly as `Decimal`, including
  comma formatting (`"1,847.50"` → `1847.50`), per-mile (`2.16`), per-cwt
  (`24.18`), sub-dollar (`0.50`), and per-each weights (`"@940lbs ea"` → `940`).

---

## The 4-category label-error taxonomy (QA portal)

Real fine-tuning data is never perfectly clean, and the demo makes that lesson
explicit. When the **QA portal** loads the seed for human review, it can *inject*
a synthetic label error into a fraction of examples so reviewers have something to
catch. Each injected error is one of four categories — the taxonomy the portal
uses to describe *how* a predicted/proposed label is wrong relative to gold:

| category         | what it means                                                                     | example                                                                 |
| ---------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **hallucinated** | A value present in the label that has **no support in the text**.                 | Emitting `reference: "PO-99999"` when no PO appears in the confirmation. |
| **missing**      | A value that **is** stated in the text but is **absent/`null`** in the label.     | Dropping `weight` when the text clearly says `4,200 lb`.                 |
| **wrong-mapping**| A stated value read into the **wrong field or wrong normalization**.              | Mapping `DLV` to `"pickup"`, or `"skids"` to `"pallets"`, or misreading `24.18` as the rate when it was the per-cwt figure. |
| **wrong-order**  | The right lines/values, but **assigned to the wrong leg or sequenced wrong** in a multi-leg confirmation. | Swapping leg 1 and leg 2 so origins/destinations line up with the wrong leg. |

These four categories are why the seed intentionally includes near-miss
temptations: partial dates (invites *missing* / *hallucinated* date parts),
abbreviations (invites *wrong-mapping* on `leg`), multi-leg relays (invites
*wrong-order*), and references that must be literal substrings (invites
*hallucinated* references). A reviewer working the QA portal is training their eye
on exactly the mistakes a fine-tuned model tends to make.

---

## Reminder

All data in this directory is **synthetic and for teaching only**. The domain is
strictly freight/logistics. Nothing here is derived from any real company's data.
