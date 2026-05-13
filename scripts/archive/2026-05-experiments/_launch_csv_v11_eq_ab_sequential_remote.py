#!/usr/bin/env python3
"""
Queue v11b to start after v11a completes (same GPU box): evidence-gated auto, then rulescored.
"""
from __future__ import annotations

import shlex
import subprocess

REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
ADAPTER = "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v4_numeric_baseline/adapter"
OUT_A = "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v11a_eq_evidence"
OUT_B = "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v11b_eq_rulescore"
WAIT = "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v11a_eq_evidence/metrics.json"

COMMON = (
    f"--adapter-in {ADAPTER} "
    "--train-csv /home/jovyan/work/train.csv "
    "--limit -1 --eval-size 600 --epochs 1 --max-length 512 "
    "--per-device-batch-size 4 --grad-accum 1 --learning-rate 8e-5 "
    "--dataloader-num-workers 8 "
    "--inject-numeric-baseline auto "
    "--inject-cipher-length-hint auto "
    "--inject-encrypt-lexical-hint auto "
    "--inject-binary-arrow-hint auto "
)

INNER = (
    f"mkdir -p {OUT_A} && cd /home/jovyan/work/Nemotron-training && "
    "bash scripts/run_v100_csv_train_4gpu.sh "
    + COMMON
    + f"--inject-equation-shape-hint auto "
    + f"--output-dir {OUT_A} "
    + f">> {OUT_A}/run.log 2>&1; "
    + f"while [ ! -f {WAIT} ]; do sleep 30; done; "
    + f"mkdir -p {OUT_B} && cd /home/jovyan/work/Nemotron-training && "
    "bash scripts/run_v100_csv_train_4gpu.sh "
    + COMMON
    + "--inject-equation-shape-hint auto_rulescored "
    + f"--output-dir {OUT_B} "
    + f">> {OUT_B}/run.log 2>&1"
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
    print(f"Queued A then B. A: tail -f {OUT_A}/run.log", flush=True)
    print(f"Then B: tail -f {OUT_B}/run.log", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
