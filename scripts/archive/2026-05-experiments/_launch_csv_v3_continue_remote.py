#!/usr/bin/env python3
"""SSH helper: start detached v3 CSV continue run (avoids PowerShell quoting bugs)."""
from __future__ import annotations

import shlex
import subprocess
import sys

REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
# Use `bash -c` (not `-lc`): login shells on this image break `/home/jovyan/work/...` mounts.
INNER = (
    "mkdir -p /home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v3_continue_gpuheavy && "
    "cd /home/jovyan/work/Nemotron-training && "
    "bash scripts/run_v100_csv_train_4gpu.sh "
    "--adapter-in /home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v2_full/adapter "
    "--train-csv /home/jovyan/work/train.csv "
    "--limit -1 --eval-size 600 --epochs 1 --max-length 512 "
    "--per-device-batch-size 4 --grad-accum 1 --learning-rate 1e-4 "
    "--dataloader-num-workers 8 "
    "--output-dir /home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v3_continue_gpuheavy "
    ">> /home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v3_continue_gpuheavy/run.log 2>&1"
)


def main() -> int:
    # One remote argv: otherwise sshd splits words and `bash -c` only sees the first token.
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
        f"  ssh {REMOTE} docker exec {CONTAINER} tail -f "
        "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v3_continue_gpuheavy/run.log",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
