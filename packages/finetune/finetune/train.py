"""Laptop-scale LoRA fine-tuning (transformers + peft + trl), CPU-only.

This actually trains: it LoRA-fine-tunes one of the registered 0.5B models on the
prepared alpaca set in a minute or two on a CPU, and writes an adapter to
``data/adapters/<model>/``. It is the tiny, honest stand-in for the GPU/Axolotl
path shipped in ``configs/`` (see quantize.md and the README's
"what's simplified" section).

Run:  uv run python -m finetune.train --model newer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from freight_schema import build_prompt
from freight_schema.paths import adapter_dir, training_dir

from finetune.registry import get_spec


def _count_hint(n: int) -> str:
    """The natural-language guidance a user would type for an n-line message."""
    return f"there are {n} shipment{'s' if n != 1 else ''} in this message"


def _load_alpaca_as_prompt_completion(
    train_file: Path, max_train: int | None, guidance_frac: float = 0.35, seed: int = 0
):
    """Read alpaca JSONL and shape it into prompt/completion pairs.

    We compose the prompt with the shared ``build_prompt`` so the string the model
    trains on is exactly the string serving will send. trl masks the prompt tokens
    and computes loss only on the completion (the JSON).

    Guidance augmentation: for a fraction of rows we put a true count hint
    ("there are N shipments in this message") into the prompt. This is what makes
    the playground's natural-language guidance box actually *work* — the model
    learns to obey a count hint, so a user can steer how many line items come back.
    Guidance steers content; the constrained decoder still guarantees the shape.
    """
    import random

    from datasets import Dataset

    rng = random.Random(seed)
    with train_file.open() as fh:
        examples = [json.loads(line) for line in fh if line.strip()]
    if max_train is not None:
        examples = examples[:max_train]

    rows = []
    for ex in examples:
        n = len(json.loads(ex["output"]))
        # The alpaca `output` is the bare JSON array (the public ParsedConfirmation
        # form). Serving constrained-decodes into the DecodeConfirmation *object*
        # ({"items": [...]}) because outlines can't target a top-level array. Train
        # on that same object form so the model's logits line up with the decoder.
        completion = '{"items": ' + ex["output"] + "}"
        # Guidance augmentation, done IN PLACE (one row per example — never drops a
        # base example, so it costs nothing in accuracy or time): put a true count
        # hint into the prompt for every multi-line example (teaching "there are N
        # shipments" -> N items) and for a fraction of single-line ones.
        use_hint = n > 1 or rng.random() < guidance_frac
        guidance = _count_hint(n) if use_hint else None
        rows.append({"prompt": build_prompt(ex["input"], guidance), "completion": completion})
    return Dataset.from_list(rows)


def train_model(
    name: str,
    train_file: Path | None = None,
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    seq_len: int = 1024,
    max_train: int | None = None,
    guidance_frac: float = 0.35,
    seed: int = 0,
) -> Path:
    """Fine-tune the registered model ``name`` and save its adapter. Returns the
    adapter directory."""
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch.set_num_threads(os.cpu_count() or 4)
    spec = get_spec(name)
    train_file = train_file or (training_dir() / "train.jsonl")
    out_dir = adapter_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_alpaca_as_prompt_completion(
        train_file, max_train, guidance_frac=guidance_frac, seed=seed
    )
    print(f"[train:{name}] base={spec.base}  rows={len(dataset)}  epochs={epochs}")

    tokenizer = AutoTokenizer.from_pretrained(spec.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec.base, dtype=torch.float32)

    # LoRA with the production *shape* (r=32, alpha=16, dropout=0.05) but narrowed
    # targets (q_proj, v_proj) for CPU speed. Production targeted all attention +
    # MLP projections (see configs/lora.yaml: q,k,v,o,up,down,gate).
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        max_length=seq_len,
        packing=False,
        completion_only_loss=True,   # loss only on the JSON, not the prompt
        use_cpu=True,                # fp32 on CPU; no GPU anywhere
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",          # we save once at the end via save_model
        report_to=[],
        seed=seed,
        disable_tqdm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    t0 = time.time()
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[train:{name}] done in {time.time() - t0:.1f}s -> {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="newer", choices=["older", "newer", "lightweight"])
    p.add_argument("--train-file", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-train", type=int, default=None,
                   help="Cap rows (keeps the CPU demo fast).")
    p.add_argument("--guidance-frac", type=float, default=0.35,
                   help="Fraction of rows trained with a count hint (teaches the "
                        "guidance box to steer line count).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    train_model(
        name=args.model,
        train_file=args.train_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        seq_len=args.seq_len,
        max_train=args.max_train,
        guidance_frac=args.guidance_frac,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
