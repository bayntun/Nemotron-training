#!/usr/bin/env python3
"""
Run holdout eval inside the Jupyter container **without vLLM** (Nemotron-H is not
supported by vLLM on many stacks).

Uses ``scripts/eval_nemotron_holdout_transformers.py`` (4bit + PEFT + same grader as
``eval.greedy_harness`` intent).

Uses a **flat** ``docker exec … python3 /abs/path/scripts/….py …`` argv list (no
``bash -lc``) so SSH does not mangle quoting; ``cd`` into the repo is not required.

By default runs **blocking** over SSH (session must stay up for the full job).

Use ``--detach`` to start the eval in the container background, optionally
``--wait`` + ``--pull-to`` to poll until the JSONL is complete and copy it to a
local path (good for symbolic-only runs from a laptop).
"""
from __future__ import annotations

import argparse
import posixpath
import shlex
import subprocess
import sys
import time
from pathlib import Path

REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
REMOTE_REPO = "/home/jovyan/work/Nemotron-training"


def _expected_jsonl_lines(subset: str) -> int:
    if subset == "symbolic-equations":
        return 9
    return 29


def _eval_argv(
    repo: str,
    adapter: str,
    val_jsonl: str,
    out_jsonl: str,
    max_new_tokens: int,
    subset: str,
) -> list[str]:
    script = posixpath.join(repo, "scripts/eval_nemotron_holdout_transformers.py")
    return [
        "python3",
        script,
        "--adapter",
        adapter,
        "--val-jsonl",
        val_jsonl,
        "--out-jsonl",
        out_jsonl,
        "--max-new-tokens",
        str(max_new_tokens),
        "--subset",
        subset,
    ]


def _ssh_docker_exec(
    inner: list[str],
    *,
    detach: bool,
) -> list[str]:
    cmd: list[str] = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=30",
        REMOTE,
        "docker",
        "exec",
    ]
    if detach:
        cmd.append("-d")
    cmd.extend(["-w", "/tmp", CONTAINER])
    cmd.extend(inner)
    return cmd


def _remote_jsonl_line_count(out_jsonl: str) -> int:
    r = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=30",
            REMOTE,
            "docker",
            "exec",
            CONTAINER,
            "wc",
            "-l",
            out_jsonl,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        return -1
    parts = r.stdout.strip().split()
    if not parts:
        return 0
    try:
        return int(parts[0])
    except ValueError:
        return -1


def _pull_jsonl(out_jsonl: str, local_path: Path) -> None:
    r = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=30",
            REMOTE,
            "docker",
            "exec",
            CONTAINER,
            "cat",
            out_jsonl,
        ],
        capture_output=True,
        check=False,
    )
    if r.returncode != 0:
        err = r.stderr.decode(errors="replace") if isinstance(r.stderr, bytes) else r.stderr
        print(err, file=sys.stderr)
        raise SystemExit(f"pull failed rc={r.returncode}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(r.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--adapter-dir",
        type=str,
        default=f"{REMOTE_REPO}/outputs/nemotron_sft_deepseek_forward_v2",
    )
    ap.add_argument(
        "--val-jsonl",
        type=str,
        default=f"{REMOTE_REPO}/data/cache/nemotron_sft_deepseek/val_greedy.jsonl",
    )
    ap.add_argument(
        "--out-jsonl",
        type=str,
        default=f"{REMOTE_REPO}/outputs/nemotron_sft_deepseek_forward_v2/eval_holdout.jsonl",
    )
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument(
        "--subset",
        choices=("all", "symbolic-equations"),
        default="all",
    )
    ap.add_argument(
        "--detach",
        action="store_true",
        help="docker exec -d (SSH returns immediately; eval keeps running in the container).",
    )
    ap.add_argument(
        "--wait",
        action="store_true",
        help="After --detach, poll remote JSONL line count until expected rows or timeout.",
    )
    ap.add_argument(
        "--pull-to",
        type=Path,
        default=None,
        help="After a successful --wait, copy the remote JSONL to this local file (binary-safe).",
    )
    ap.add_argument("--poll-seconds", type=int, default=90)
    ap.add_argument("--timeout-seconds", type=int, default=10800)
    args = ap.parse_args()

    argv = _eval_argv(
        REMOTE_REPO,
        args.adapter_dir,
        args.val_jsonl,
        args.out_jsonl,
        args.max_new_tokens,
        args.subset,
    )

    if args.detach:
        subprocess.run(_ssh_docker_exec(argv, detach=True), check=True)
        print(
            f"Started detached eval in {CONTAINER} (flat docker exec -d; no shell).\n"
            f"Monitor progress: JSONL line count should reach {_expected_jsonl_lines(args.subset)}.\n"
            f"  ssh {REMOTE} docker exec {CONTAINER} wc -l {shlex.quote(args.out_jsonl)}",
            flush=True,
        )
        if not args.wait:
            return 0
        want = _expected_jsonl_lines(args.subset)
        deadline = time.monotonic() + args.timeout_seconds
        while time.monotonic() < deadline:
            n = _remote_jsonl_line_count(args.out_jsonl)
            print(f"… remote lines={n} (want {want})", flush=True)
            if n >= want:
                break
            time.sleep(args.poll_seconds)
        else:
            print("ERROR: timeout waiting for JSONL rows", file=sys.stderr)
            return 1
        if args.pull_to:
            _pull_jsonl(args.out_jsonl, args.pull_to)
            print(f"Pulled to {args.pull_to}", flush=True)
        print(f"Eval finished. Remote file: {args.out_jsonl}", flush=True)
        return 0

    subprocess.run(_ssh_docker_exec(argv, detach=False), check=True)
    print(f"Eval finished. See {args.out_jsonl} on {CONTAINER}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
