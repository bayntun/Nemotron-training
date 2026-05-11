#!/usr/bin/env python3
"""Stream a small tar of CSV-trainer files into remote docker (no git on server)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
DEST = "/home/jovyan/work/Nemotron-training"
FILES = [
    "tmp_train_csv_remote.py",
    "train/csv_numeric_baseline.py",
    "train/csv_cipher_length_hint.py",
    "train/test_csv_numeric_baseline.py",
    "train/test_csv_cipher_length_hint.py",
    "scripts/run_v100_csv_train_4gpu.sh",
]


def main() -> int:
    recv = subprocess.Popen(
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
        stdin=subprocess.PIPE,
    )
    assert recv.stdin is not None
    send = subprocess.Popen(
        ["tar", "-cf", "-"] + FILES,
        cwd=REPO,
        stdout=recv.stdin,
    )
    send.wait()
    recv.stdin.close()
    recv.wait()
    if send.returncode != 0 or recv.returncode != 0:
        print(f"tar send={send.returncode} recv={recv.returncode}", file=sys.stderr)
        return 1
    print(f"Synced into {CONTAINER}:{DEST}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
