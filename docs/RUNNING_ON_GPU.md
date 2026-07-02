# Running on a GPU — scaling the demo to a production-tier model

> **Handoff for the GPU/H100 session.** The laptop demo (`make demo`) trains and
> serves *real* open-source models — Qwen2-0.5B and Qwen2.5-0.5B — with genuine
> LoRA fine-tuning (`transformers`+`peft`+`trl`) and genuine constrained-decoding
> inference (`outlines`). Nothing is mocked. It's just **small and CPU-bound**, so
> exact-match accuracy sits around a few percent (a 0.5B genuinely struggles to
> reproduce dates and reference strings verbatim; the per-field breakdown still
> shows the newer base beating the older).
>
> On a real GPU (e.g. an H100) you can run the **8B-class production tier** from
> the talk and get the ~90%+ exact-match the slides quote — **same pipeline, same
> schema, same playground, same benchmark gate.** This doc is the checklist to get
> there. Keep the CPU defaults untouched so `git clone && make demo` still works on
> any laptop; the GPU model is an *additional* path for the live demo.

## The one-paragraph mental model

Everything is parameterized by the model **registry**
(`packages/finetune/finetune/models.toml`) and env overrides. Point a registry
slot at a bigger base, fine-tune it (via `train.py` with GPU settings, or the
production Axolotl config), and serve it — the rest of the code doesn't change,
because the schema, prompt, constrained-decoding target, reference guard, and
DuckDB gate are all model-agnostic.

## 0. Prereqs on the GPU box

```bash
git clone <repo> && cd freight-parser
uv sync
# Ensure a CUDA build of torch (uv may resolve the CPU wheel on Linux):
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu124
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 1. Generate the training data (same as the laptop path)

```bash
uv run python packages/qa_portal/manage.py migrate
uv run python packages/qa_portal/manage.py seed --auto-review
uv run python packages/qa_portal/manage.py export_training --out data/training/qa_real.jsonl
uv run python -m synthetic.cli generate --n 2000 --seed 0      # scale up freely on a GPU
uv run python -m synthetic.cli export --out data/training/synthetic.jsonl
uv run python -m finetune.prep_dataset --holdout-count 100 --force   # bigger, less-noisy benchmark
```

On a GPU you want a **larger benchmark** (100–300) so exact-match is stable, and
you can drop the `--max-train` cap entirely (train on everything).

## 2. Pick a bigger model — two options

### Option A — quick, via `train.py` (recommended to start)

Point the `newer` slot at an 8B instruct base and let the existing trainer run it
with GPU-appropriate settings. No code change needed:

```bash
export FREIGHT_MODEL_NEWER="Qwen/Qwen2.5-7B-Instruct"   # or Llama-3.1-8B-Instruct, etc.
uv run python -m finetune.train --model newer \
    --epochs 3 --batch-size 8 --seq-len 1024          # no --max-train => full set
```

Then widen the LoRA and use bf16 for real runs. The knobs live in
`packages/finetune/finetune/train.py::train_model`:
- `target_modules`: change `["q_proj","v_proj"]` → all seven
  (`q,k,v,o_proj, up,down,gate_proj`) to match production.
- In the `SFTConfig`: set `bf16=True`, `use_cpu=False`, bump
  `per_device_train_batch_size`, and consider `packing=True`.

(These are intentionally narrowed for CPU in the shipped code — the comments say
so. Widen them here; don't commit the change to the default CPU path.)

### Option B — production fidelity, via Axolotl

The Axolotl configs already target an 8B and encode the full production recipe
(flash attention, `adamw_bnb_8bit`, all target modules, `sample_packing`, 4
epochs). This is the "real" path referenced throughout the repo.

```bash
uv pip install --python .venv axolotl        # intentionally NOT a default dep
# edit packages/finetune/configs/lora.yaml -> set base_model + output_dir to
# data/adapters/newer, confirm the dataset path is data/training/train.jsonl
axolotl train packages/finetune/configs/lora.yaml
# (qlora.yaml is the 4-bit variant for a smaller consumer GPU)
```

See `packages/finetune/quantize.md` for the merge-then-quantize deployment note.

## 3. Benchmark + serve — unchanged

```bash
uv run python -m eval.predict  --model newer     # and --model older for the comparison
uv run python -m eval.report                      # the DuckDB release-gate table
make serve                                        # playground at http://localhost:8000/
```

The serving layer loads the trained adapter for whatever base the registry points
at and constrained-decodes against the same schema. The playground, model
dropdown, guidance box, and reference guard all work identically.

## 4. What changes, what doesn't

| | CPU laptop demo | GPU / H100 |
|---|---|---|
| Base model | 0.5B | 8B-class |
| Trainer | TRL+PEFT, fp32, `q,v`, ~2 epochs, capped rows | TRL bf16 all-modules, or Axolotl |
| Exact-match | ~a few % (illustrative) | ~90%+ (the talk's numbers) |
| Everything else | — | **identical** (schema, prompt, outlines, DuckDB gate, playground) |

## 5. Gotchas already handled for you

- **`Decimal` breaks outlines' DFA builder** (pydantic emits a lookahead regex).
  The repo decodes against a `float`+`Literal` mirror schema (`DecodeConfirmation`)
  and re-validates into exact `Decimal`. This holds at any model size — don't
  "simplify" it back to Decimal.
- **Train/serve prompt parity** comes from the shared `build_prompt`; keep using
  it (the trainer already does).
- **Guidance→item-count** is trained in via `train.py`'s count-hint augmentation;
  a bigger model will follow it far more precisely than the 0.5B.

## 6. Suggested flow for the live talk

1. Open the CPU demo's playground first ("this runs on your laptop — `git clone &&
   make demo`"), show valid JSON coming out of the 0.5B.
2. Switch to the H100-served 8B (same playground, `FREIGHT_MODEL_NEWER` pointed at
   the 8B + its adapter) and re-parse the same messy confirmation — same endpoint,
   dramatically better content.
3. Show the DuckDB comparison table with the real numbers as the "release gate."
