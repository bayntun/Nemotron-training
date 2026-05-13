#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--id", default="", help="If set, print only this row id")
    args = ap.parse_args()

    n = 0
    with args.csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p = str(row.get("prompt", ""))
            if args.id and str(row.get("id", "")) != args.id:
                continue
            pl = p.lower()
            if "transformation rules" not in pl and "equation" not in pl:
                continue
            print(f"--- id={row.get('id')}")
            print(p[:1200].replace("\n", "\\n"))
            n += 1
            if args.id:
                break
            if n >= args.limit:
                break
    print(f"shown={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
