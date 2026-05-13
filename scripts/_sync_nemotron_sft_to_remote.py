#!/usr/bin/env python3
"""Stream Nemotron SFT entrypoints + configs into the remote Jupyter container."""
from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
DEST = "/home/jovyan/work/Nemotron-training"
FILES = [
    "eval/symbolic_holdout.py",
    "eval/greedy_harness.py",
    "train/sft.py",
    "train/_dataset.py",
    "train/_lora.py",
    "configs/accelerate_zero2.yaml",
    "configs/accelerate_single_gpu_fp16.yaml",
    "configs/accelerate_multigpu_fp16.yaml",
    "configs/deepspeed_zero2.json",
    "scripts/build_nemotron_sft_jsonl_from_deepseek_synth.py",
    "scripts/build_eval_jsonl_from_nemotron_train.py",
    "scripts/run_nemotron_sft_deepseek_synth_4gpu.sh",
    "scripts/push_nemotron_sft_dataset_remote.py",
    "scripts/_launch_nemotron_sft_deepseek_remote.py",
    "scripts/_run_greedy_eval_nemotron_holdout_remote.py",
    "scripts/eval_nemotron_holdout_transformers.py",
    "scripts/report_symbolic_holdout_eval.py",
    "scripts/summarize_eval_failures.py",
    "scripts/analyze_val_prompt_mix.py",
]


def _file_bytes(rel: str) -> bytes:
    raw = (REPO / rel).read_bytes()
    if rel.endswith(".sh"):
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return raw


def _tar_stream() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for rel in FILES:
            data = _file_bytes(rel)
            info = tarfile.TarInfo(name=rel.replace("\\", "/"))
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def main() -> int:
    payload = _tar_stream()
    recv = subprocess.run(
        [
            "ssh",
            REMOTE,
            "docker",
            "exec",
            "-i",
            CONTAINER,
            "tar",
            "xf",
            "-",
            "-C",
            DEST,
        ],
        input=payload,
        check=False,
    )
    if recv.returncode != 0:
        print(f"remote tar extract failed rc={recv.returncode}", file=sys.stderr)
        return 1
    print(f"Synced Nemotron SFT bundle into {CONTAINER}:{DEST}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
