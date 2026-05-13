#!/usr/bin/env python3
"""
SSH helper: continue TinyLlama CSV LoRA from v2_full adapter (~39% first-token / ~30% full)
with --inject-numeric-baseline auto (linear m→m′ + gravity d=k·t² hints).

Requires repo on the server to include tmp_train_csv_remote.py with --inject-numeric-baseline.
Uses bash -c (not -lc) so /home/jovyan/work paths resolve inside the container.
"""
from __future__ import annotations

import shlex
import subprocess
import sys

REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
OUT = "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v4_numeric_baseline"
ADAPTER = "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v2_full/adapter"

INNER = (
    f"mkdir -p {OUT} && cd /home/jovyan/work/Nemotron-training && "
    "bash scripts/run_v100_csv_train_4gpu.sh "
    f"--adapter-in {ADAPTER} "
    "--train-csv /home/jovyan/work/train.csv "
    "--limit -1 --eval-size 600 --epochs 1 --max-length 512 "
    "--per-device-batch-size 4 --grad-accum 1 --learning-rate 1e-4 "
    "--dataloader-num-workers 8 "
    "--inject-numeric-baseline auto "
    f"--output-dir {OUT} "
    f">> {OUT}/run.log 2>&1"
)


def main() -> int:
    remote = f"docker exec -d {CONTAINER} bash -c {shlex.quote(INNER)}"
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=30",
        REMOTE,
        remote,
    ]
    print("Running:", "ssh", REMOTE, "docker exec -d ...", file=sys.stderr)
    subprocess.run(cmd, check=True)
    print(
        "Detached job started. Tail log:\n"
        f"  ssh {REMOTE} docker exec {CONTAINER} tail -f {OUT}/run.log",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
