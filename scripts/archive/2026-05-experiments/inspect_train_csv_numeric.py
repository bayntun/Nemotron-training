#!/usr/bin/env python3
"""Print sample prompts whose answers look numeric (for manual pattern review)."""
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

FLOATISH = re.compile(r"^[-+]?[0-9]+\.[0-9]+$")
INTISH = re.compile(r"^[-+]?[0-9]+$")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=Path, nargs="?", default=Path("/home/jovyan/work/train.csv"))
    p.add_argument("--n", type=int, default=30, help="How many random numeric rows to print")
    args = p.parse_args()
    rows: list[dict[str, str]] = []
    with args.csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    cand: list[dict[str, str]] = []
    for row in rows:
        a = (row.get("answer") or "").strip()
        if FLOATISH.match(a) or INTISH.match(a):
            cand.append(row)
    print(f"total_rows={len(rows)} numeric_answer_rows={len(cand)}")
    random.seed(0)
    sample = random.sample(cand, min(args.n, len(cand)))
    for row in sample:
        prompt = (row.get("prompt") or "").strip().replace("\n", " ")
        if len(prompt) > 280:
            prompt = prompt[:277] + "..."
        print("---")
        print("answer:", (row.get("answer") or "").strip())
        print("prompt:", prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
