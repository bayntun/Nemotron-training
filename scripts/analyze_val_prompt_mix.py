#!/usr/bin/env python3
"""
Summarize ``val_greedy.jsonl`` (or any eval input JSONL) for **what to target** without running the model.

Prints counts by ``category``, ground-truth length buckets, and simple prompt keyword hits
(equation / decrypt / cipher / binary / numeric patterns).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("val_jsonl", type=Path)
    args = ap.parse_args()
    if not args.val_jsonl.is_file():
        print(f"ERROR: missing {args.val_jsonl}", file=sys.stderr)
        return 2

    cats: Counter[str] = Counter()
    lens: Counter[str] = Counter()
    kw: dict[str, int] = {
        "equation_style": 0,
        "decrypt_cipher": 0,
        "binary_arrow": 0,
        "wonderland": 0,
        "numeric_table": 0,
    }
    n = 0
    with args.val_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            cats[str(r.get("category", "unknown"))] += 1
            gt = str(r.get("ground_truth", ""))
            L = len(gt)
            if L <= 8:
                lens["1-8"] += 1
            elif L <= 32:
                lens["9-32"] += 1
            elif L <= 128:
                lens["33-128"] += 1
            else:
                lens["129+"] += 1
            p = str(r.get("prompt", "")).lower()
            if "=" in p and ("determine" in p or "result for" in p):
                kw["equation_style"] += 1
            if "decrypt" in p or "cipher" in p or "encryption" in p:
                kw["decrypt_cipher"] += 1
            if re.search(r"\b[01]{4,}\s*->", p):
                kw["binary_arrow"] += 1
            if "wonderland" in p or "alice" in p:
                kw["wonderland"] += 1
            if re.search(r"\d+\.\d+|\d+\s+\d+", p) and "example" in p:
                kw["numeric_table"] += 1

    print(f"rows={n}\n")
    print("by category:")
    for k, v in cats.most_common():
        print(f"  {k!r}: {v}")
    print("\nby ground_truth length:")
    for k in ["1-8", "9-32", "33-128", "129+"]:
        print(f"  {k}: {lens[k]}")
    print("\nkeyword-style buckets (overlapping):")
    for k, v in kw.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
