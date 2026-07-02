"""FastAPI serving layer + browser playground.

Thin HTTP wrapper over :mod:`serving.inference`. The heavy lifting — constrained
decoding, the Decimal round-trip, the reference guard, model caching — lives in
``inference`` so the evaluator can share it. This module is just routes + the
single-page playground served at ``/``.

Run:  uv run uvicorn serving.app:app --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from freight_schema import dump_lines
from finetune.registry import DEFAULT_MODEL
from serving.inference import available_models, parse

_STATIC = Path(__file__).with_name("static")

app = FastAPI(
    title="freight-parser",
    description="Self-hosted, fine-tuned LLM that parses freight shipment "
    "confirmations into structured JSON.",
    version="0.1.0",
)


class ParseRequest(BaseModel):
    """Request body for ``POST /parse``. ``guidance`` and ``model`` are optional."""

    text: str
    guidance: str | None = None
    model: str | None = None


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """No favicon to serve — return 204 so the browser console stays clean."""
    from fastapi import Response

    return Response(status_code=204)


@app.get("/ready")
def ready() -> dict:
    """Liveness/readiness probe."""
    return {"status": "ready"}


@app.get("/models")
def models() -> dict:
    """List registry models, flagging which have a trained adapter + the default."""
    return {"models": available_models(), "default": DEFAULT_MODEL}


@app.post("/parse")
def parse_endpoint(req: ParseRequest) -> JSONResponse:
    """Parse a confirmation into structured JSON.

    Returns numbers as JSON numbers (Decimals preserved on the way in, rendered
    naturally on the way out). ``guidance`` steers content; the constrained
    decoder still guarantees the shape.
    """
    result = parse(req.text, guidance=req.guidance, model=req.model)
    # dump_lines emits numbers-as-numbers; json.loads gives plain dicts so FastAPI
    # doesn't re-serialize Decimals as strings.
    items = json.loads(dump_lines(result.confirmation.root))
    return JSONResponse(
        {
            "items": items,
            "model": result.model,
            "requested_model": result.requested_model,
            "fell_back": result.fell_back,
            "reference_guarded": result.reference_guarded,
            "duration": round(result.duration, 3),
        }
    )


# The playground is a static single-page app. Mount it LAST and at the root, with
# html=True so `/` serves index.html and its relative assets (app.js, style.css)
# resolve. Registering it after the API routes means /ready, /models, and /parse
# are matched first; everything else falls through to the static files.
app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="playground")
