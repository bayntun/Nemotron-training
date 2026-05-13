#!/usr/bin/env python3
"""
Build a ``--val-jsonl`` file for ``eval.greedy_harness`` from Nemotron SFT ``train.jsonl``.

Each input line has ``prompt`` and ``answer`` (full teacher CoT). Ground truth for scoring
is ``extract_final_answer(answer)`` (same boxed extraction as the competition grader).

This measures how well the base model + LoRA reproduces the **final** answer after greedy
decode — useful when you do not yet have a disjoint holdout split. For unbiased accuracy,
reserve a slice of the DeepSeek synth JSONL at **dataset build time** and train only on
the complement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.grader import extract_final_answer


def _row_id(prompt: str, idx: int) -> str:
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{idx:05d}-{h}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train-jsonl",
        type=Path,
        default=REPO / "data" / "cache" / "nemotron_sft_deepseek" / "train.jsonl",
        help="Nemotron SFT JSONL (prompt + answer teacher text).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL path (greedy_harness schema).",
    )
    ap.add_argument("--every-n", type=int, default=1, help="Emit every Nth row (1 = all).")
    ap.add_argument("--max-rows", type=int, default=0, help="Stop after this many emitted rows (0 = no cap).")
    ap.add_argument(
        "--category",
        type=str,
        default="nemotron_train_boxed",
        help="Category label stored in each record.",
    )
    args = ap.parse_args()

    if not args.train_jsonl.is_file():
        print(f"ERROR: missing {args.train_jsonl}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    seen = 0
    with args.train_jsonl.open(encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not prompt or not answer:
                continue
            if seen % args.every_n != 0:
                seen += 1
                continue
            seen += 1
            gt = extract_final_answer(answer)
            if not gt or gt == "NOT_FOUND":
                continue
            rec = {
                "id": _row_id(prompt, emitted),
                "prompt": prompt,
                "ground_truth": gt,
                "category": args.category,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            emitted += 1
            if args.max_rows and emitted >= args.max_rows:
                break

    print(f"wrote {emitted} rows -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
