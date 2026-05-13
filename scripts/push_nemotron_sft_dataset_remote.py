#!/usr/bin/env python3
"""SCP Nemotron SFT JSONL files to the host, then docker cp into the Jupyter container."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
CONTAINER_DIR = "/home/jovyan/work/Nemotron-training/data/cache/nemotron_sft_deepseek"


def _push_one(local: Path, remote_tmp: str, container_path: str) -> None:
    if not local.is_file():
        print(f"ERROR: missing {local}", file=sys.stderr)
        raise SystemExit(2)
    subprocess.run(["scp", str(local.resolve()), f"{REMOTE}:{remote_tmp}"], check=True)
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", REMOTE, "docker", "exec", CONTAINER, "mkdir", "-p", CONTAINER_DIR],
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
            remote_tmp,
            f"{CONTAINER}:{container_path}",
        ],
        check=True,
    )
    print(f"Pushed {local} -> {CONTAINER}:{container_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train-jsonl",
        type=Path,
        default=_REPO / "data" / "cache" / "nemotron_sft_deepseek" / "train.jsonl",
    )
    ap.add_argument(
        "--val-jsonl",
        type=Path,
        default=None,
        help="Optional val_greedy.jsonl (same schema as greedy_harness).",
    )
    args = ap.parse_args()

    _push_one(args.train_jsonl, "/tmp/nemotron_sft_train.jsonl", f"{CONTAINER_DIR}/train.jsonl")
    if args.val_jsonl is not None:
        _push_one(
            args.val_jsonl,
            "/tmp/nemotron_sft_val_greedy.jsonl",
            f"{CONTAINER_DIR}/val_greedy.jsonl",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
