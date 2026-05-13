#!/usr/bin/env python3
"""SCP merged train CSV to the SSH host, then docker cp into the Jupyter container."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
REMOTE_TMP = "/tmp/train_merged_deepseek_full.csv"
CONTAINER_PATH = "/home/jovyan/work/train_merged_deepseek_full.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "local_csv",
        type=Path,
        nargs="?",
        default=_REPO / "data" / "cache" / "train_remote_merged_deepseek_full.csv",
        help="Local merged CSV path",
    )
    args = ap.parse_args()
    src = args.local_csv.resolve()
    if not src.is_file():
        print(f"ERROR: missing {src}", file=sys.stderr)
        return 2

    subprocess.run(
        ["scp", str(src), f"{REMOTE}:{REMOTE_TMP}"],
        check=True,
    )
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            REMOTE,
            "docker",
            "cp",
            REMOTE_TMP,
            f"{CONTAINER}:{CONTAINER_PATH}",
        ],
        check=True,
    )
    print(f"Pushed {src} -> {CONTAINER}:{CONTAINER_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
