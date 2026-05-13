#!/usr/bin/env python3
"""
v13: Same stack as v12 rulescore, but train on DeepSeek-merge CSV (238 ids overwritten).

Requires ``train_merged_deepseek_full.csv`` in the Jupyter container under
``/home/jovyan/work/`` (upload from ``data/cache/train_remote_merged_deepseek_full.csv``).
"""
from __future__ import annotations

import shlex
import subprocess

REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
OUT = "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v13_deepseek_merged"
ADAPTER = "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v4_numeric_baseline/adapter"
TRAIN_CSV = "/home/jovyan/work/train_merged_deepseek_full.csv"

INNER = (
    f"mkdir -p {OUT} && cd /home/jovyan/work/Nemotron-training && "
    "bash scripts/run_v100_csv_train_4gpu.sh "
    f"--adapter-in {ADAPTER} "
    f"--train-csv {TRAIN_CSV} "
    "--limit -1 --eval-size 600 --epochs 1 --max-length 512 "
    "--per-device-batch-size 4 --grad-accum 1 --learning-rate 8e-5 "
    "--dataloader-num-workers 8 "
    "--inject-numeric-baseline auto "
    "--inject-cipher-length-hint auto "
    "--inject-encrypt-lexical-hint auto "
    "--inject-equation-shape-hint auto_rulescored "
    "--inject-binary-arrow-hint auto "
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
