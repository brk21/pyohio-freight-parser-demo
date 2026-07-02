# From adapter to a lean server: merge, then quantize

> **TL;DR** — Fine-tuning gives you a tiny LoRA *adapter* sitting on top of a
> big frozen base. To ship it, you (1) **merge** the adapter back into the base
> to get one clean set of weights, then (2) **quantize** those weights to 4-bit
> so the server needs a fraction of the memory. This page is conceptual — you
> do **not** need any of it for the CPU demo, which loads the base + adapter
> directly and runs in float on your laptop.

## Where this fits in the freight-parser story

The laptop demo (`make demo`) fine-tunes a small model on CPU and serves it as
`base weights + LoRA adapter`, loaded straight from disk. That's perfect for a
talk: nothing to merge, nothing to quantize, no GPU.

But picture the **production** version from the Axolotl configs in this folder
(`lora.yaml` / `qlora.yaml`): an 8B-class model that parses freight shipment
confirmations into structured JSON, sitting behind a dispatch tool that hits it
thousands of times a day. Now memory and latency matter, and you want the
smallest, fastest artifact that still parses "3 skids PU Columbus OH, DLV
Cincinnati, rate 850" correctly. That's what the two steps below buy you.

## Step 1 — Merge the adapter into the base

A LoRA adapter is a set of small low-rank matrices that *add a correction* to
the base model's weights. During training the base is frozen and only the
adapter learns. At inference you have two choices:

- **Keep them separate** (what the demo does): load base, load adapter, apply
  the correction on the fly. Flexible — you can hot-swap adapters — but you're
  carrying two objects and paying a small runtime cost to combine them.
- **Merge** (what production does): fold the adapter's correction permanently
  into the base weights, producing one standalone model. Nothing to combine at
  request time; it loads and behaves like any ordinary model.

Conceptually the merge is just `W_merged = W_base + (B · A) · scaling`, done
once, offline. With PEFT it's a one-liner (`merge_and_unload()`); the output is
a normal set of full-precision weights.

Merge when the adapter is final and you want a single deployable artifact.
Stay unmerged while you're still iterating on adapters or want to serve several
freight variants (say, one tuned for LTL confirmations and one for full
truckload) off the same base.

## Step 2 — Quantize the merged weights for serving

The merged model is still full precision (fp16/bf16) — an 8B model is ~16 GB.
**Quantization** stores each weight in fewer bits so the whole thing gets
dramatically smaller and cheaper to run, at the cost of a little numerical
precision.

- **4-bit / NF4** is the common serving target. NF4 ("NormalFloat 4") is a
  4-bit format shaped to match how neural-net weights are actually distributed,
  so it keeps more useful signal than a naive 4-bit rounding. Optionally
  *double quantization* also compresses the per-block scale factors for a bit
  more savings. An ~16 GB model drops to roughly ~5 GB.
- For our narrow, well-structured task — text in, JSON out — the accuracy hit
  from 4-bit is typically negligible: the model still emits the same
  `origin` / `destination` / `quantity` / `rate` fields. Always re-run the
  benchmark after quantizing to confirm the freight fields still match.

> **Merge vs. QLoRA — don't confuse them.** `qlora.yaml` quantizes the base to
> 4-bit *during training* to fit a smaller GPU. The quantization here is a
> separate, *post-training/deployment* step applied to the **merged** model so
> the server is lean. You can train full-precision LoRA and still quantize for
> serving, or train with QLoRA and then merge+re-quantize — they're independent
> decisions.

## The whole path at a glance

```
  base model (frozen, 8B, fp16)
        │  fine-tune with LoRA / QLoRA   (lora.yaml / qlora.yaml, on GPU)
        ▼
  base + small LoRA adapter              ← what the CPU demo serves, in float
        │  Step 1: merge_and_unload()
        ▼
  one merged fp16 model
        │  Step 2: 4-bit / NF4 quantize
        ▼
  compact quantized model                → deploy behind the freight parser API
```

## Why you can ignore all of this for the demo

On a CPU laptop there's no VRAM budget to defend and no thousands-per-day
traffic — so the demo keeps the base and adapter separate and runs in plain
float. Merging and quantizing are the moves you make when the freight parser
graduates from "runs in a conference talk" to "runs in the dispatch stack."
Same model, same JSON schema — just packaged for the road.
