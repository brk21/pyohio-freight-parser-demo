"""Serving API tests.

Fast tests monkeypatch the (heavy) inference core so the HTTP layer is covered
without loading a model. The model-dependent behaviors the talk demos — valid
ParsedConfirmation out, and a guidance hint changing the line count — run only
when FREIGHT_MODEL_TESTS=1 (i.e. after `make demo` has trained adapters), so the
default `make test` stays fast and offline.
"""

import os
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from freight_schema import ParsedConfirmation, ShipmentLine
from serving import app as app_module
from serving.inference import ParseResult

client = TestClient(app_module.app)

MODEL_TESTS = os.environ.get("FREIGHT_MODEL_TESTS") == "1"
requires_model = pytest.mark.skipif(
    not MODEL_TESTS, reason="set FREIGHT_MODEL_TESTS=1 (after `make demo`) to run"
)


def test_ready():
    assert client.get("/ready").json() == {"status": "ready"}


def test_models_lists_older_newer_and_default():
    data = client.get("/models").json()
    names = {m["name"] for m in data["models"]}
    assert {"older", "newer"} <= names
    assert data["default"] == "newer"
    # every model advertises whether it has a trained adapter
    assert all("has_adapter" in m for m in data["models"])


def test_parse_shape_monkeypatched(monkeypatch):
    def fake_parse(text, guidance=None, model=None):
        line = ShipmentLine(quantity=2, unit="pallets", leg="pickup",
                            rate=Decimal("1847.50"), reference=None)
        return ParseResult(
            confirmation=ParsedConfirmation(root=[line]),
            model="newer", requested_model=model or "newer",
            fell_back=False, reference_guarded=True, duration=0.42,
        )

    monkeypatch.setattr(app_module, "parse", fake_parse)
    resp = client.post("/parse", json={"text": "PU 2 pallets CHI->LAX 1847.50"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["items"][0]["rate"] == 1847.5  # number, not string
    assert body["items"][0]["quantity"] == 2
    assert body["model"] == "newer"
    assert body["reference_guarded"] is True


@requires_model
def test_parse_returns_valid_confirmation():
    resp = client.post("/parse", json={"text": "PU 12 pallets CHI -> LAX rate 1847.50"})
    body = resp.json()
    # must validate against the canonical schema
    ParsedConfirmation.model_validate(body["items"])
    assert len(body["items"]) >= 1


@requires_model
def test_guidance_hint_yields_requested_line_count():
    # A clearly-delimited 3-leg confirmation. Guidance steers *content* (how many
    # line items); the constrained decoder still guarantees the shape.
    text = (
        "Leg 1: PU 20 pallets CHI to ATL 1500.00\n"
        "Leg 2: DLV 10 skids ATL to DFW 800.00\n"
        "Leg 3: PU 5 cartons DFW to LAX 300.00"
    )
    hinted = client.post(
        "/parse", json={"text": text, "guidance": "there are 3 shipments in this message"}
    ).json()
    assert len(hinted["items"]) == 3
    # sanity: still schema-valid
    from freight_schema import ParsedConfirmation
    ParsedConfirmation.model_validate(hinted["items"])
