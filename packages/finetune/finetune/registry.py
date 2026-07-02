"""The switchable-model registry + shared model loading.

This module is the single place that knows how to turn a friendly model name
(``"older"`` / ``"newer"`` / ``"lightweight"``) into a Hugging Face base id, an
adapter directory on disk, and a ready-to-run merged model. Both the serving
layer and the evaluator import from here so they load models identically.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from freight_schema.paths import adapter_dir

_REGISTRY_TOML = Path(__file__).with_name("models.toml")

# The model served when a request doesn't name one. Overridable via env.
DEFAULT_MODEL = os.environ.get("FREIGHT_DEFAULT_MODEL", "newer")


@dataclass(frozen=True)
class ModelSpec:
    """One registered base model."""

    name: str          # registry key: "older" | "newer" | "lightweight"
    base: str          # Hugging Face model id
    generation: str    # e.g. "prior gen" / "current gen" (for the UI/table)
    label: str         # display label, e.g. "newer-0.5b"
    note: str          # short description

    @property
    def adapter_path(self) -> Path:
        """Where this model's trained LoRA adapter lives on disk."""
        return adapter_dir(self.name)

    @property
    def has_adapter(self) -> bool:
        """True once ``train.py`` has written an adapter for this model."""
        return (self.adapter_path / "adapter_config.json").exists()

    @property
    def is_default(self) -> bool:
        return self.name == DEFAULT_MODEL


@lru_cache(maxsize=1)
def load_registry() -> dict[str, ModelSpec]:
    """Parse ``models.toml``, applying ``FREIGHT_MODEL_<NAME>`` env overrides."""
    with _REGISTRY_TOML.open("rb") as fh:
        raw = tomllib.load(fh)
    specs: dict[str, ModelSpec] = {}
    for name, cfg in raw.get("models", {}).items():
        env_override = os.environ.get(f"FREIGHT_MODEL_{name.upper()}")
        specs[name] = ModelSpec(
            name=name,
            base=env_override or cfg["base"],
            generation=cfg.get("generation", ""),
            label=cfg.get("label", name),
            note=cfg.get("note", ""),
        )
    return specs


def get_spec(name: str) -> ModelSpec:
    """Look up a :class:`ModelSpec` by registry name (raises if unknown)."""
    registry = load_registry()
    if name not in registry:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(registry)}")
    return registry[name]


def model_names() -> list[str]:
    """Registry keys in file order."""
    return list(load_registry())


# ---------------------------------------------------------------------------
# Model loading (shared by training, serving, and eval).
# ---------------------------------------------------------------------------


def load_base_and_tokenizer(base_id: str):
    """Load a base causal-LM + tokenizer for CPU (fp32).

    Kept import-local so merely importing the registry (e.g. to list models in
    the API) doesn't pull in torch/transformers.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32)
    return model, tokenizer


def load_for_inference(name: str):
    """Return an eval-ready ``(model, tokenizer)`` for the named model.

    If a trained adapter exists it is merged into the base weights (which is what
    ``outlines`` wants — a plain model, not a PEFT wrapper). If not, the bare base
    is returned so the pipeline still runs before any training has happened.
    """
    spec = get_spec(name)
    model, tokenizer = load_base_and_tokenizer(spec.base)
    if spec.has_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(spec.adapter_path)).merge_and_unload()
    return model.eval(), tokenizer
