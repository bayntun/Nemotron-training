#!/usr/bin/env python3
"""
v8b: v4 adapter + full hint stack + assistant-only CE loss + binary hints, with lower LR and 2 epochs.
"""
from __future__ import annotations

import shlex
import subprocess

REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
OUT = "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v8b_assistant_lowlr"
ADAPTER = "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v4_numeric_baseline/adapter"

INNER = (
    f"mkdir -p {OUT} && cd /home/jovyan/work/Nemotron-training && "
    "bash scripts/run_v100_csv_train_4gpu.sh "
    f"--adapter-in {ADAPTER} "
    "--train-csv /home/jovyan/work/train.csv "
    "--limit -1 --eval-size 600 --epochs 2 --max-length 512 "
    "--per-device-batch-size 4 --grad-accum 1 --learning-rate 2e-5 "
    "--dataloader-num-workers 8 "
    "--inject-numeric-baseline auto "
    "--inject-cipher-length-hint auto "
    "--inject-encrypt-lexical-hint auto "
    "--inject-equation-shape-hint auto "
    "--inject-binary-arrow-hint auto "
    "--assistant-loss-only "
    f"--output-dir {OUT} "
    f">> {OUT}/run.log 2>&1"
)


def main() -> int:
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=30",
            REMOTE,
            f"docker exec -d {CONTAINER} bash -c {shlex.quote(INNER)}",
        ],
        check=True,
    )
    print(f"Detached. tail -f {OUT}/run.log", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
