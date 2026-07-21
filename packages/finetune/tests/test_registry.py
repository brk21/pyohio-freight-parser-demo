"""Registry decoding-contract tests.

These are deliberately dependency-light: they exercise ``_force_greedy`` with a
fake generation_config object, so they need neither torch/transformers nor a
model download. Importing ``finetune.registry`` is cheap because it keeps
torch/transformers imports inside the loader functions.
"""

from __future__ import annotations

from types import SimpleNamespace

from finetune.registry import _force_greedy


def test_force_greedy_overrides_sampling_defaults():
    """A base's sampling generation_config must be forced to greedy.

    Instruct bases (e.g. Qwen2.5) default to do_sample=True/temperature=0.7, and
    outlines forwards generation to transformers without overriding it. If this
    regresses, served parses stop being deterministic — so pin the contract here.
    """
    gc = SimpleNamespace(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)

    returned = _force_greedy(gc)

    # greedy: sampling off, and the sampling knobs cleared so transformers does
    # not warn (and cannot re-enable sampling behavior).
    assert gc.do_sample is False
    assert gc.temperature is None
    assert gc.top_p is None
    assert gc.top_k is None
    # returns the same object for convenience
    assert returned is gc
