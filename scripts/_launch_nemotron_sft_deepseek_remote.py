#!/usr/bin/env python3
"""
Start Nemotron QLoRA SFT on DeepSeek CoT JSONL (detached, logs under the chosen output dir).

Profiles:
  forward (default) — writes to ``outputs/nemotron_sft_deepseek_forward_v2``; uses ``MAX_LEN=512``,
    ``EPOCHS=1.0``, ``GRAD_ACCUM=24``, thinking template **on** (matches ``eval.greedy_harness``),
    ``DATALOADER_NUM_WORKERS=2``, ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.
    Push ``train.jsonl`` (+ optional ``val_greedy.jsonl`` from a holdout rebuild) before launching.

  smoke — tiny run for CI / sanity (16 samples, short context).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess

REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
REPO = "/home/jovyan/work/Nemotron-training"

PROFILES: dict[str, dict[str, str]] = {
    "smoke": {
        "out": f"{REPO}/outputs/nemotron_sft_deepseek_smoke",
        "env": (
            "export ACCELERATE_CONFIG=configs/accelerate_single_gpu_fp16.yaml && "
            "export NEMOTRON_NO_THINKING=1 && "
            "MAX_SAMPLES=16 EPOCHS=0.1 MAX_LEN=256 GRAD_ACCUM=8 "
        ),
    },
    "forward": {
        "out": f"{REPO}/outputs/nemotron_sft_deepseek_forward_v2",
        "env": (
            "export ACCELERATE_CONFIG=configs/accelerate_single_gpu_fp16.yaml && "
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
            "export DATALOADER_NUM_WORKERS=2 && "
            "EPOCHS=1.0 MAX_LEN=512 GRAD_ACCUM=24 "
        ),
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="forward",
        help="smoke = tiny sanity; forward = full jsonl + longer context (default).",
    )
    ap.add_argument(
        "--print-only",
        action="store_true",
        help="Print the remote bash snippet instead of launching.",
    )
    args = ap.parse_args()
    prof = PROFILES[args.profile]
    out = prof["out"]
    inner = (
        f"mkdir -p {out} && cd {REPO} && "
        f"{prof['env']}"
        f"OUTPUT_DIR={out} "
        "bash scripts/run_nemotron_sft_deepseek_synth_4gpu.sh "
        f">> {out}/run.log 2>&1"
    )
    if args.print_only:
        print(inner)
        return 0
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=30",
            REMOTE,
            f"docker exec -d {CONTAINER} bash -c {shlex.quote(inner)}",
        ],
        check=True,
    )
    print(f"Detached Nemotron SFT ({args.profile}). tail -f {out}/run.log", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
