#!/usr/bin/env python3
"""Print wrong cases from a greedy-style eval JSONL (``is_correct`` field)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_jsonl", type=Path)
    ap.add_argument("--max-print", type=int, default=50)
    args = ap.parse_args()
    if not args.eval_jsonl.is_file():
        print(f"ERROR: missing {args.eval_jsonl}", file=sys.stderr)
        return 2

    total = ok = 0
    fails: list[dict] = []
    with args.eval_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            if r.get("is_correct"):
                ok += 1
            else:
                fails.append(r)

    print(f"total={total} correct={ok} wrong={len(fails)} acc={ok / total if total else 0:.4f}\n")
    for j, r in enumerate(fails[: args.max_print]):
        pid = r.get("id", "")
        cat = r.get("category", "")
        gt = str(r.get("ground_truth", ""))[:200]
        ex = str(r.get("extracted", ""))[:200]
        pr = str(r.get("prompt", ""))[:400].replace("\n", " ")
        print(f"--- failure {j + 1} id={pid} category={cat!r}")
        print(f"  ground_truth: {gt!r}")
        print(f"  extracted:    {ex!r}")
        print(f"  prompt_head:  {pr!r}…")
        print()
    if len(fails) > args.max_print:
        print(f"(… {len(fails) - args.max_print} more failures not shown; raise --max-print)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
