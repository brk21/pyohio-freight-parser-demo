# synthetic — label-first freight-confirmation generator

Generates messy, realistic-looking freight **shipment confirmations** paired with
their exact structured labels, then parks them in a local DuckDB store for the
training-prep stage to read.

The domain is freight/logistics only, and every carrier name, lane code, and
commodity is invented and generic — clean-room synthetic data.

## The core idea: label first, text second

Real annotation is "read text → write label", which is slow and error-prone.
Here we invert it. We **build a fully structured
[`ShipmentLine`](../schema/freight_schema/models.py) first** by sampling from a
generic freight vocabulary, then render freetext *from* that record. Because the
text is derived from the label — never the reverse — the gold JSON is correct
**by construction**. No parsing, no heuristics, no drift between text and label.

```
sample fields ─▶ ShipmentLine(s) ─▶ render freetext ─▶ {text, parsed_json}
                     (the label)        (the input)      correct by construction
```

## Render styles

| style      | looks like                                                    |
| ---------- | ------------------------------------------------------------- |
| `terse`    | `PU 12 plt CHI-LAX 15,400# 4/5 1,847.50 PO-4471`              |
| `verbose`  | full-sentence confirmation email with carrier + commodity     |
| `abbrev`   | abbreviation-heavy, label-tagged: `PU 12 PLT ORIG DFW ... RT` |
| `multileg` | multi-stop load confirmation; every leg repeats the reference |

## Complexity knobs

A style plus a complexity level (1–`MAX_COMPLEXITY`) is turned into concrete
[`Knobs`](synthetic/generator.py):

- **number of legs** — 1, or 2–4 for `multileg`
- **missing-field probability** — how often optional fields (weight, lane, date,
  reference, accessorial) are dropped
- **abbreviation density** — chance of a short surface form over a spelled-out one
- **partial-date style** — how aggressively dates truncate to month / month+day /
  full, never fabricating a part the text does not state

## Determinism

Everything is a pure function of a seed. `generate_examples(n, seed)` threads one
`random.Random(seed)` through the whole run, so the same `(n, seed)` always
produces byte-identical output.

## Invariants (enforced by construction + tests)

- Every `parsed_json` validates as `list[ShipmentLine]`.
- A reference stored on a line is a **literal substring** of the text (so the
  serving-time reference guard never nulls a correct one).
- Multi-leg confirmations render every leg and repeat the reference on each.
- Dates keep only the parts actually stated.

## The store

`store.py` writes to the DuckDB file at `freight_schema.paths.synthetic_db()`.
*In production this role was a cloud data warehouse; DuckDB is the offline laptop
stand-in.* Table:

```
examples(id VARCHAR, text VARCHAR, parsed_json VARCHAR, style VARCHAR, complexity INTEGER)
```

`parsed_json` is the canonical JSON **string** of the gold lines. Export emits
JSONL of `{"id", "text", "parsed_json", "style"}`, the shared hand-off format.

## CLI

```bash
# generate + (re)store in DuckDB
uv run python -m synthetic.cli generate --n 500 --seed 0
# or via the console script
uv run freight-synthetic generate --n 500 --seed 0

# peek at a few stored rows
uv run python -m synthetic.cli query

# export to JSONL (default: data/training/synthetic.jsonl; relative paths
# resolve against the repo root)
uv run python -m synthetic.cli export --out data/training/synthetic.jsonl
```

## Tests

```bash
uv run pytest packages/synthetic -q
```
