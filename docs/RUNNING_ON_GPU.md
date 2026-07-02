# Running on a GPU — scaling the demo to a production-tier model

> **Runbook for the GPU/H100 session.** The laptop demo (`make demo`) trains and
> serves *real* open-source models — Qwen2-0.5B and Qwen2.5-0.5B — with genuine
> LoRA fine-tuning (`transformers`+`peft`+`trl`) and genuine constrained-decoding
> inference (`outlines`). Nothing is mocked. It's just **small and CPU-bound**.
>
> On a real GPU (e.g. an H100) you run the **8B-class production tier** from the
> talk and get high exact-match — **same pipeline, same schema, same playground,
> same benchmark gate.** This doc is the step-by-step runbook, verified end to end
> on an H100 PCIe (80 GB). Everything here is **opt-in behind `FREIGHT_GPU=1`**, so
> `git clone && make demo` still works byte-for-byte on any laptop; the 8B is an
> *additional* path for the live demo.

## The one-paragraph mental model

Everything is parameterized by the model **registry**
(`packages/finetune/finetune/models.toml`) plus a handful of env vars. The `older`
and `newer` slots are the two 0.5B generations (the laptop story). A third
`production` slot points at an 8B-class base (`Qwen/Qwen2.5-7B-Instruct`). You
fine-tune all three with LoRA, benchmark them into one DuckDB table, and serve the
8B as the default — the rest of the code doesn't change, because the schema,
prompt, constrained-decoding target (`DecodeConfirmation`), reference guard, and
release gate are all model-agnostic.

## The env-var contract (this is the whole GPU switch)

| Env var | Effect | Set it for |
|---|---|---|
| `FREIGHT_GPU=1` | Opt into the GPU recipe: **fp32+TF32** training (bf16 diverges — see gotchas) and **bf16 CUDA** inference. Unset ⇒ the CPU laptop default, unchanged. | all GPU train/eval/serve |
| `FREIGHT_LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj` | Widen LoRA to all attention+MLP projections. Default (unset) is the narrow `q_proj,v_proj`. | the **8B only** — leave UNSET for the 0.5Bs (see gotcha 2) |
| `TORCHDYNAMO_DISABLE=1` | Force eager execution (skips a `torch.compile`→triton build that fails on missing `Python.h`). | GPU **eval + serve** |
| `FREIGHT_DEFAULT_MODEL=production` | Serve the 8B as the playground default. | GPU **serve** |
| `FREIGHT_MAX_NEW_TOKENS=512` | Decode headroom for long multi-leg confirmations (default 384). | GPU serve (optional) |

## 0. Prereqs on the GPU box

```bash
git clone <repo> && cd freight-parser
uv sync
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> CUDA: True NVIDIA H100 PCIe
```

`uv sync` resolved a CUDA torch wheel automatically here (`torch 2.12.1+cu130`); no
manual reinstall was needed. If yours resolves the CPU wheel instead, force it:
`uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu124`.

## 1. Generate the training data

```bash
uv run python packages/qa_portal/manage.py migrate --noinput
uv run python packages/qa_portal/manage.py seed --auto-review
uv run python packages/qa_portal/manage.py export_training --out data/training/qa_real.jsonl
uv run python -m synthetic.cli generate --n 2000 --seed 0
uv run python -m synthetic.cli export --out data/training/synthetic.jsonl
uv run python -m finetune.prep_dataset --synthetic-ratio 40 --holdout-count 150 --force
# -> 58 real + 2000 synthetic -> 1908 train, 150 benchmark
```

> **Why `--synthetic-ratio 40`?** prep mixes `ratio × (#real)` synthetic rows. With
> only 58 real rows, the default ratio 4 caps synthetic at ~232 and wastes the
> 2000 you generated. A high ratio uses them all. `--holdout-count 150` gives a
> bigger, more stable benchmark than the laptop demo's 24.

## 2. Fine-tune the 8B (`production`) — widened LoRA, fp32+TF32

```bash
export FREIGHT_GPU=1 TORCHDYNAMO_DISABLE=1
export FREIGHT_LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
uv run python -m finetune.train --model production \
    --epochs 3 --batch-size 4 --grad-accum 4 --seq-len 1024 --lr 2e-4
# ~11 min, ~37 GB VRAM on an H100; final train_loss ~0.03
```

The adapter lands in `data/adapters/production/`. To use a different 8B, either
add a registry entry or override the base: `export FREIGHT_MODEL_PRODUCTION=...`
(Qwen2.5-7B-Instruct is Apache-2.0 and ungated; Llama-3.1-8B-Instruct is a fine
drop-in if you accept its license/gating).

## 3. Fine-tune the two 0.5Bs — **narrow** LoRA (the generational comparison)

```bash
unset FREIGHT_LORA_TARGETS          # <- CRITICAL: 0.5Bs use the narrow q,v LoRA
export FREIGHT_GPU=1 TORCHDYNAMO_DISABLE=1
uv run python -m finetune.train --model older --epochs 2 --batch-size 8 --grad-accum 1 --lr 2e-4
uv run python -m finetune.train --model newer --epochs 2 --batch-size 8 --grad-accum 1 --lr 2e-4
# ~80 s each. (Equivalent: `uv run python -m finetune.train_all --epochs 2`, slower batch.)
```

Widening the LoRA on the 0.5Bs saturates them (both ~98% — the generational gap
disappears). Keeping the narrow `q_proj,v_proj` adapter is what lets the newer
base pull ahead. See gotcha 2.

## 4. Benchmark + serve

```bash
export FREIGHT_GPU=1 TORCHDYNAMO_DISABLE=1
uv run python -m eval.predict --model older
uv run python -m eval.predict --model newer
uv run python -m eval.predict --model production
uv run python -m eval.report                       # the DuckDB release-gate table

FREIGHT_GPU=1 TORCHDYNAMO_DISABLE=1 FREIGHT_DEFAULT_MODEL=production FREIGHT_MAX_NEW_TOKENS=512 \
    uv run uvicorn serving.app:app --host 0.0.0.0 --port 8000
# playground at http://localhost:8000/  (first parse per model warms it + compiles the FSM)
```

**Reference result (150 held-out, seeds above):**

| model | base | pct_correct | pct_fields |
|---|---|---:|---:|
| `production-7b` | Qwen2.5-7B-Instruct | **99.3** | 99.9 |
| `newer-0.5b` | Qwen2.5-0.5B-Instruct | 88.0 | 98.9 |
| `older-0.5b` | Qwen2-0.5B-Instruct | 79.3 | 98.0 |

## 5. What changes, what doesn't

| | CPU laptop demo | GPU / H100 |
|---|---|---|
| Default model | 0.5B (`newer`) | 8B (`production`, via `FREIGHT_DEFAULT_MODEL`) |
| Trainer dtype | fp32, `use_cpu`, `q,v` LoRA, ~2 epochs, capped rows | **fp32+TF32**, all-modules LoRA (8B), 3 epochs, full set |
| Inference | fp32 on CPU | bf16 on CUDA |
| Exact-match | a few % (data-starved 0.5B) | ~99% (8B) |
| Everything else | — | **identical** (schema, prompt, outlines, reference guard, DuckDB gate, playground) |

## 6. Gotchas (all handled — here's why the recipe looks the way it does)

- **`Decimal` breaks outlines' DFA builder** (pydantic emits a lookahead regex).
  The repo decodes against a `float`+`Literal` mirror (`DecodeConfirmation`) and
  re-validates into exact `Decimal`. Holds at any model size — don't "simplify" it.
- **Train/serve prompt parity** comes from the shared `build_prompt`; keep using it.
- **bf16 LoRA SFT diverges on this data** — deterministically, at ~0.4 epoch: the
  forward loss stays finite but the *backward* gradient goes `NaN` once the model
  gets overconfident (a bf16 backward-precision failure). Fix: train in **fp32 +
  TF32** on the GPU (`bf16=False, tf32=True`, weights loaded fp32). It's stable and
  still ~2 s/step for the 7B on an H100. Inference is unaffected (forward-only), so
  it still serves in bf16.
- **Widen LoRA on the 8B only.** On the 0.5Bs the widened adapter has enough
  capacity to memorize the (same-distribution) benchmark, so both hit ~98% and the
  "newer base wins for free" story vanishes. The narrow `q,v` adapter is
  capacity-limited, so the newer generation's stronger base shows through.
- **`torch.compile` fails on this box** (`TORCHDYNAMO_DISABLE=1` fixes it). GPU
  generation triggers a triton runtime gcc build that can't find `Python.h` (no
  python-dev headers). `torch.compile` is only a perf optimization; forcing eager
  is correct and fast enough.

## 7. Production-fidelity alternative — Axolotl

`packages/finetune/configs/lora.yaml` encodes the full production recipe (flash
attention, `adamw_bnb_8bit`, `sample_packing`, all target modules, 4 epochs) and
already targets an 8B. It's intentionally **not** a default dependency.

```bash
uv pip install --python .venv axolotl        # separate GPU env; drags in CUDA-only wheels
# point base_model + output_dir at data/adapters/production, dataset -> data/training/train.jsonl
axolotl train packages/finetune/configs/lora.yaml
```

See `packages/finetune/quantize.md` for the merge-then-quantize deployment note.

## 8. Suggested flow for the live talk

1. Open the CPU demo's playground first ("this runs on your laptop — `git clone &&
   make demo`"), show valid JSON coming out of the 0.5B.
2. Switch to the H100-served 8B (`production`) and re-parse the same messy
   confirmation — same endpoint, dramatically better content.
3. Show the DuckDB comparison table as the release gate. Never train live on stage.
