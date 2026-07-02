# freight-qa-portal

A tiny Django app that turns the model's raw auto-parses into **trusted training
labels**. It is the human-in-the-loop stage of the freight-parser pipeline: a
reviewer opens each freetext shipment confirmation next to the model's parsed
JSON, fixes what's wrong, and signs off. Only signed-off records become training
data.

Everything is laptop-local — SQLite on disk (`packages/qa_portal/db.sqlite3`),
no external services, and the Django admin *is* the whole UI (mounted at `/`).

## The review cockpit

The portal is the Django admin, tuned for triage. On the `Confirmation`
changelist you get:

- a one-line **text preview** of each confirmation,
- **reviewed?** / **corrected?** / **exclude?** columns,
- filters on **reviewed**, **exclude**, and a **corrected** (yes/no) filter,
- full-text **search** over the confirmation text.

Open a record to see three editable pieces:

| field       | meaning                                                            |
|-------------|--------------------------------------------------------------------|
| `text`      | the raw confirmation, exactly as received (read as context)        |
| `parsed`    | the model's auto-parse — a JSON list of `ShipmentLine` records      |
| `corrected` | your fix (same shape); leave blank if the auto-parse was already OK |
| `exclude`   | check to keep a junk example **out** of training                   |

Every save is validated against the shared `freight_schema` (`ParsedConfirmation`)
in `Confirmation.clean()`, so you can never persist JSON that would break
downstream training — a bad `parsed` or `corrected` is rejected with a
field-level error.

The record's **training label** is `corrected` if you supplied one, otherwise
`parsed`. That single rule (see `Confirmation.training_label`) is what the export
uses.

### "Save & next" flow

The change form adds a **"Save & next (mark reviewed)"** button next to the
usual Save buttons. Clicking it:

1. saves any correction you typed,
2. stamps the record `reviewed = now()`, and
3. redirects you straight to the **next unreviewed** confirmation.

So you march through the queue without bouncing back to the changelist. When the
queue is empty it returns you to the (now-cleared) list. Records you never touch
stay `reviewed = None` and remain in the queue.

## The four QA error classes

The `seed` command deliberately injects one of each of the four error classes
the talk teaches, so the cockpit has real mistakes to catch. The true gold is
always known, so each corrupted record can be fixed back to correct:

| class                   | what it looks like                                   |
|-------------------------|------------------------------------------------------|
| **(a) hallucinated field** | a value that is *not* in the text (e.g. an invented PO/BOL `reference`) |
| **(b) missing information** | a present value dropped from the parse (e.g. a stated `weight` set to null) |
| **(c) incorrect mapping**   | the right field with the wrong value (e.g. a wrong `quantity`) |
| **(d) wrong ordering**      | multi-leg lines shuffled out of order (the lines are right, their order isn't) |

`seed` picks a distinct, suitable record for each class (by shape, since the
seed data is authored elsewhere) and prints which record demonstrates which.

## Commands

All run from the repo root via the workspace venv:

```bash
# create the SQLite schema
uv run python packages/qa_portal/manage.py migrate

# load seed confirmations + inject the 4 error classes (reviewer has work to do)
uv run python packages/qa_portal/manage.py seed

# ...or simulate a completed QA pass (every record reviewed, corrupted ones fixed)
uv run python packages/qa_portal/manage.py seed --auto-review

# create a local reviewer login, then open the cockpit
uv run python packages/qa_portal/manage.py createsuperuser
uv run python packages/qa_portal/manage.py runserver   # -> http://127.0.0.1:8000/

# export the trusted labels as training JSONL
uv run python packages/qa_portal/manage.py export_training --out data/training/qa_real.jsonl
```

### `seed`

Loads `data/seed/confirmations.jsonl` (path from `freight_schema.paths.seed_file()`),
setting `parsed` to each row's gold. Injects the four error classes into a
handful of records. Idempotent — existing confirmations are cleared first.

- default: corrupted records are left **unreviewed** with no `corrected`, so a
  live reviewer has genuine fixes to make.
- `--auto-review`: stamps **every** record `reviewed = now()` and sets
  `corrected = gold` on the corrupted ones — a completed QA pass, ready to
  export without opening the UI.

### `export_training`

Emits the QA export format the rest of the pipeline consumes:

```json
{"id": "1", "text": "PU 12 pallets CHI->LAX ...", "parsed_json": "[{\"origin\": \"CHI\", ...}]"}
```

Rules: only rows with `reviewed` set, `exclude=True` rows skipped, label is
`corrected` over `parsed`, and `parsed_json` is canonical
(`freight_schema.dump_lines`). Defaults to `data/training/qa_real.jsonl`
(`freight_schema.paths.training_dir()`).

## Tests

```bash
uv run pytest packages/qa_portal -q
```

Django is configured without `pytest-django`: `tests/conftest.py` calls
`django.setup()` and builds an in-memory SQLite test DB. Tests cover schema
validation in `clean()` and the `export_training` selection rules / output
validity.
