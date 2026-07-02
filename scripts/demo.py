#!/usr/bin/env python
"""Run the whole freight-parser pipeline end to end — the `make demo` entrypoint.

Five stages, one closed loop, on a CPU laptop in a few minutes:

  1. QA portal   — migrate + seed + auto-review (simulating completed QA), then
                   export the trusted labels.
  2. Synthetic   — generate messy confirmations into DuckDB and export them.
  3. Prep        — merge real + synthetic into an alpaca set, hold out a benchmark.
  4. Fine-tune   — LoRA-train the older AND newer 0.5B models (1 epoch each).
  5. Benchmark   — score both on the holdout; print the per-category breakdown and
                   the older-vs-newer comparison table (the release gate).
  6. Serve       — boot the API, parse the same example with each model to show
                   the difference, print the playground URL, and leave it running.

Each stage runs as a subprocess (exactly the commands a user would type), so a
failure surfaces immediately and the whole script exits non-zero. Sizes are kept
small so the loop finishes fast on CPU; override with the flags below.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable  # the uv-managed venv interpreter
MANAGE = str(ROOT / "packages" / "qa_portal" / "manage.py")

# A messy multi-line sample used for the live older-vs-newer contrast at the end.
DEMO_SAMPLE = (
    "ORION LOGISTICS  REF BOL-5521\n"
    "DLV 500 cartons JUN 27 ATL->DFW 0.50/lb\n"
    "PU 300 skids SEP 24 ATL->DFW 2.16\n"
    "DLV 120 pallets 24.18"
)

# Child processes: quiet, deterministic, CPU-only.
CHILD_ENV = {
    **os.environ,
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONWARNINGS": "ignore",
    "PYTHONUNBUFFERED": "1",
}


def banner(n: int | str, title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  STAGE {n}: {title}\n{bar}", flush=True)


def run_step(cmd: list[str], desc: str) -> float:
    """Run a subprocess, stream its output, time it, and raise on failure."""
    print(f"$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), env=CHILD_ENV)
    dt = time.time() - t0
    if result.returncode != 0:
        raise SystemExit(f"\n[FAILED] {desc} (exit {result.returncode}) after {dt:.1f}s")
    print(f"[ok] {desc} ({dt:.1f}s)", flush=True)
    return dt


def http_get(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def http_post(url: str, payload: dict, timeout: float = 180.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(base: str, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if http_get(f"{base}/ready", timeout=2.0).get("status") == "ready":
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic-n", type=int, default=200, help="synthetic rows to generate")
    p.add_argument("--holdout", type=int, default=24, help="benchmark holdout size")
    p.add_argument("--max-train", type=int, default=110, help="cap training rows (CPU speed)")
    p.add_argument("--epochs", type=int, default=2,
                   help="epochs per model (2 keeps the tiny models from over-segmenting)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--keep-alive", action=argparse.BooleanOptionalAction, default=True,
                   help="leave the server running at the end (default). --no-keep-alive "
                        "runs the two demo parses then exits 0 (for CI/verification).")
    args = p.parse_args(argv)

    overall = time.time()
    timings: dict[str, float] = {}

    # --- Stage 1: QA portal -------------------------------------------------
    banner(1, "Training-data QA (Django + Pydantic)")
    timings["qa.migrate"] = run_step([PY, MANAGE, "migrate", "--noinput"], "migrate QA portal DB")
    timings["qa.seed"] = run_step([PY, MANAGE, "seed", "--auto-review"],
                                  "seed + auto-review (simulating completed QA)")
    qa_real = "data/training/qa_real.jsonl"
    timings["qa.export"] = run_step([PY, MANAGE, "export_training", "--out", qa_real],
                                    "export reviewed labels")

    # --- Stage 2: Synthetic data -------------------------------------------
    banner(2, "Synthetic data (Python + DuckDB)")
    timings["synth.gen"] = run_step(
        [PY, "-m", "synthetic.cli", "generate", "--n", str(args.synthetic_n), "--seed", str(args.seed)],
        "generate synthetic confirmations",
    )
    synth = "data/training/synthetic.jsonl"
    timings["synth.export"] = run_step([PY, "-m", "synthetic.cli", "export", "--out", synth],
                                       "export synthetic set")

    # --- Stage 3: Dataset prep ---------------------------------------------
    banner(3, "Dataset prep (alpaca: real + synthetic, holdout)")
    timings["prep"] = run_step(
        [PY, "-m", "finetune.prep_dataset", "--real", qa_real, "--synthetic", synth,
         "--synthetic-ratio", "4", "--holdout-count", str(args.holdout),
         "--max-train", str(args.max_train), "--seed", str(args.seed), "--force"],
        "prepare training + benchmark sets",
    )

    # --- Stage 4: Fine-tune both models ------------------------------------
    banner(4, "Fine-tune older + newer (LoRA on CPU)")
    timings["train"] = run_step(
        [PY, "-m", "finetune.train_all", "--epochs", str(args.epochs),
         "--max-train", str(args.max_train), "--seed", str(args.seed)],
        "LoRA fine-tune older and newer",
    )

    # --- Stage 5: Benchmark -------------------------------------------------
    banner(5, "Benchmark (DuckDB release gate)")
    for model in ("older", "newer"):
        timings[f"eval.{model}"] = run_step([PY, "-m", "eval.predict", "--model", model],
                                            f"benchmark {model}")
    timings["eval.report"] = run_step([PY, "-m", "eval.report"], "cross-model comparison")

    # --- Stage 6: Serve + live contrast ------------------------------------
    banner(6, "Serve + playground (FastAPI + outlines)")
    base = f"http://127.0.0.1:{args.port}"
    server = subprocess.Popen(
        [PY, "-m", "uvicorn", "serving.app:app", "--host", "127.0.0.1", "--port", str(args.port),
         "--log-level", "warning"],
        cwd=str(ROOT), env=CHILD_ENV,
    )
    try:
        if not wait_ready(base):
            server.terminate()
            raise SystemExit("[FAILED] serving app did not become ready")
        print(f"[ok] server ready at {base}", flush=True)

        print("\nSame confirmation, each model (constrained decoding => always valid JSON):")
        print("-" * 72)
        print(DEMO_SAMPLE)
        print("-" * 72)
        for model in ("older", "newer"):
            t0 = time.time()
            resp = http_post(f"{base}/parse", {"text": DEMO_SAMPLE, "model": model})
            print(f"\n[{model}]  ({time.time()-t0:.1f}s, model used: {resp['model']})")
            print(json.dumps(resp["items"], indent=2))
            if resp.get("reference_guarded"):
                print("  note: reference guard nulled a fabricated PO/BOL value")

        total = time.time() - overall
        print("\n" + "=" * 72)
        print(f"  DEMO COMPLETE in {total:.0f}s")
        print("  Stage timings: " + ", ".join(f"{k}={v:.0f}s" for k, v in timings.items()))
        print(f"\n  ▶ Playground:  {base}/")
        print("     (paste text, switch older<->newer, try a guidance hint)")
        print("=" * 72, flush=True)

        if args.keep_alive:
            print("\nServer is running. Press Ctrl-C to stop.\n", flush=True)
            server.wait()
        else:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
    except KeyboardInterrupt:
        print("\nStopping server…", flush=True)
        server.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
