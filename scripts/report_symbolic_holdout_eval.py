#!/usr/bin/env python3
"""
Summarize eval JSONL for symbolic-equation holdout rows only.

Reads lines like ``eval_nemotron_holdout_transformers.py`` output (id, prompt,
ground_truth, extracted, is_correct, …), keeps prompts that match
``eval.symbolic_holdout.is_symbolic_equation_holdout_prompt``, prints accuracy
and per-row correct / wrong (expected vs extracted).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.symbolic_holdout import is_symbolic_equation_holdout_prompt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-jsonl", type=Path, required=True)
    args = ap.parse_args()

    if not args.eval_jsonl.is_file():
        print(f"ERROR: missing {args.eval_jsonl}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    with args.eval_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    sym = [r for r in rows if is_symbolic_equation_holdout_prompt(str(r.get("prompt") or ""))]
    n = len(sym)
    if n == 0:
        print("No symbolic-equation rows in eval file (check filter or run full eval).", file=sys.stderr)
        return 1

    correct = sum(1 for r in sym if r.get("is_correct"))
    acc = correct / n
    print(f"Symbolic equation holdout: {correct}/{n} = {acc:.4f}\n")

    wrong = [r for r in sym if not r.get("is_correct")]
    ok = [r for r in sym if r.get("is_correct")]

    def row_line(r: dict) -> str:
        rid = str(r.get("id", ""))
        gt = str(r.get("ground_truth", ""))
        ex = str(r.get("extracted", ""))
        return f"  id={rid}  ground_truth={gt!r}  extracted={ex!r}"

    print("Correct:")
    for r in ok:
        print(row_line(r))
    print("\nWrong:")
    if not wrong:
        print("  (none)")
    else:
        for r in wrong:
            print(row_line(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
