#!/usr/bin/env python3
"""
Continue from v4 adapter + numeric baseline, add cipher length-preserving hints.

Requires synced tmp_train_csv_remote.py (inject-cipher-length-hint) on the server.
"""
from __future__ import annotations

import shlex
import subprocess
import sys

REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
OUT = "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v5_cipher_hints"
ADAPTER = "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v4_numeric_baseline/adapter"

INNER = (
    f"mkdir -p {OUT} && cd /home/jovyan/work/Nemotron-training && "
    "bash scripts/run_v100_csv_train_4gpu.sh "
    f"--adapter-in {ADAPTER} "
    "--train-csv /home/jovyan/work/train.csv "
    "--limit -1 --eval-size 600 --epochs 1 --max-length 512 "
    "--per-device-batch-size 4 --grad-accum 1 --learning-rate 8e-5 "
    "--dataloader-num-workers 8 "
    "--inject-numeric-baseline auto "
    "--inject-cipher-length-hint auto "
    f"--output-dir {OUT} "
    f">> {OUT}/run.log 2>&1"
)


def main() -> int:
    remote = f"docker exec -d {CONTAINER} bash -c {shlex.quote(INNER)}"
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", REMOTE, remote],
        check=True,
    )
    print(
        "Detached. Tail:\n"
        f"  ssh {REMOTE} docker exec {CONTAINER} tail -f {OUT}/run.log",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
