"""Constrained-decoding inference core.

This is the beating heart of both the API and the benchmark: text in, a
schema-valid :class:`~freight_schema.ParsedConfirmation` out. It is deliberately
framework-free (no FastAPI here) so the evaluator can call the *exact same*
inference path the API serves — "same endpoint as production."

The flow:

1. ``build_prompt(text, guidance)`` — identical to the training prompt, plus an
   optional plain-English hint appended to the instruction.
2. ``outlines`` constrained decoding against :class:`DecodeConfirmation` (a
   float+Literal schema; see freight_schema for why Decimal can't be the target).
   The decoder *guarantees the JSON shape*; the fine-tuned weights supply the
   content. Greedy decoding (``do_sample=False``, forced in the model registry so
   it overrides the base's sampling generation_config) so a given input parses the
   same way on a fixed machine.
3. Re-validate the raw JSON into the canonical Decimal schema.
4. Apply the reference anti-hallucination guard.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from freight_schema import (
    DecodeConfirmation,
    ParsedConfirmation,
    apply_reference_guard,
    build_prompt,
    decode_to_confirmation,
)
from finetune.registry import DEFAULT_MODEL, get_spec, load_for_inference, load_registry

# Cap on generated tokens. A single confirmation's JSON is short; this bounds
# worst-case latency on CPU. Overridable for unusually long multi-leg messages.
MAX_NEW_TOKENS = int(os.environ.get("FREIGHT_MAX_NEW_TOKENS", "384"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.ERROR)


@dataclass
class _Loaded:
    model: object
    tokenizer: object
    generator: object


# Loaded models are cached by name so flipping older<->newer in the playground is
# instant after the first use (and the FSM/DFA is compiled only once per model).
_CACHE: dict[str, _Loaded] = {}


@dataclass
class ParseResult:
    """Everything a caller might want about one parse."""

    confirmation: ParsedConfirmation
    model: str            # the model actually used (post-fallback)
    requested_model: str  # what the caller asked for
    fell_back: bool       # True if the requested model had no adapter and we swapped
    reference_guarded: bool  # True if the guard nulled a hallucinated reference
    duration: float       # wall-clock seconds for the decode


def resolve_model(requested: str | None) -> tuple[str, bool]:
    """Pick the model to actually run.

    Falls back to the default model when the requested one is unknown or has no
    trained adapter yet, so the pipeline works before/against partial training.
    Returns ``(effective_name, fell_back)``.
    """
    registry = load_registry()
    name = requested or DEFAULT_MODEL
    if name not in registry:
        return DEFAULT_MODEL, True
    spec = registry[name]
    if not spec.has_adapter:
        default_spec = registry.get(DEFAULT_MODEL)
        if default_spec is not None and default_spec.has_adapter and name != DEFAULT_MODEL:
            return DEFAULT_MODEL, True
    return name, False


def _load(name: str) -> _Loaded:
    """Lazily load + cache a model and its constrained-JSON generator."""
    if name in _CACHE:
        return _CACHE[name]
    import outlines

    model, tokenizer = load_for_inference(name)
    om = outlines.from_transformers(model, tokenizer)
    generator = outlines.Generator(om, DecodeConfirmation)
    loaded = _Loaded(model=model, tokenizer=tokenizer, generator=generator)
    _CACHE[name] = loaded
    return loaded


def parse(text: str, guidance: str | None = None, model: str | None = None) -> ParseResult:
    """Parse one confirmation into a :class:`ParsedConfirmation`.

    ``guidance`` steers *content* (e.g. "there are 3 shipments in this message");
    the constrained decoder still guarantees the *shape*. That contrast is the
    whole point: a natural-language hint can change how many line items come
    back, but it can never produce malformed JSON.
    """
    effective, fell_back = resolve_model(model)
    loaded = _load(effective)

    prompt = build_prompt(text, guidance)
    t0 = time.time()
    # do_sample=False is redundant with the greedy generation_config set at load
    # (registry.load_base_and_tokenizer), but we pass it explicitly so the decode
    # call states its own determinism contract. outlines forwards it to
    # transformers' generate().
    raw = loaded.generator(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    duration = time.time() - t0

    confirmation = decode_to_confirmation(raw)

    # Reference guard: null out any PO/BOL number the model invented.
    before = [line.reference for line in confirmation.root]
    apply_reference_guard(confirmation.root, text)
    after = [line.reference for line in confirmation.root]
    reference_guarded = before != after

    return ParseResult(
        confirmation=confirmation,
        model=effective,
        requested_model=model or DEFAULT_MODEL,
        fell_back=fell_back,
        reference_guarded=reference_guarded,
        duration=duration,
    )


def available_models() -> list[dict]:
    """Registry contents for the ``GET /models`` endpoint / playground dropdown."""
    return [
        {
            "name": spec.name,
            "base": spec.base,
            "label": spec.label,
            "generation": spec.generation,
            "note": spec.note,
            "has_adapter": spec.has_adapter,
            "is_default": spec.is_default,
        }
        for spec in load_registry().values()
    ]


def warmup(name: str | None = None) -> None:
    """Preload a model and compile its decoder (pay the cold cost up front).

    Handy right before a live demo so the first real parse is fast.
    """
    effective, _ = resolve_model(name)
    _load(effective)
    # A trivial parse compiles/caches the FSM so later calls are warm.
    parse("PU 1 pallet CHI -> LAX rate 100.00", model=effective)
