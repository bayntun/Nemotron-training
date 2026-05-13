#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

EQ = re.compile(r"^\s*(\d{2})\s*([^\s\d])\s*(\d{2})\s*=\s*(-?\d+)\s*$")


def rev2(n: int) -> int:
    s = f"{abs(n):02d}"
    r = int(s[::-1])
    return -r if n < 0 else r


def unrev_int(n: int) -> int:
    sign = -1 if n < 0 else 1
    s = str(abs(n))
    return sign * int(s[::-1])


def cands(a: int, b: int) -> dict[str, int]:
    base = {
        "add": a + b,
        "sub": a - b,
        "mul": a * b,
        "concat": int(f"{a}{b}"),
    }
    ar, br = rev2(a), rev2(b)
    base.update(
        {
            "rev_add": ar + br,
            "rev_sub": ar - br,
            "rev_mul": ar * br,
            "rev_concat": int(f"{ar}{br}"),
            "unrev_rev_add": unrev_int(ar + br),
            "unrev_rev_sub": unrev_int(ar - br),
            "unrev_rev_mul": unrev_int(ar * br),
            "unrev_rev_mul_plus1": unrev_int(ar * br + 1),
        }
    )
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--prompt-col", default="prompt")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open("r", encoding="utf-8")))
    stats = Counter()
    op_hits: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, str]] = []

    for r in rows:
        p = str(r.get(args.prompt_col, ""))
        lines = p.splitlines()
        eqs = []
        for ln in lines:
            m = EQ.match(ln)
            if not m:
                continue
            eqs.append((int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))))
        if len(eqs) < 2:
            continue
        stats["prompts_detected"] += 1
        prompt_ok_any = False
        for a, op, b, y in eqs:
            vals = cands(a, b)
            hit = []
            for name, v in vals.items():
                if y == v:
                    hit.append(name)
                if y == v + 1:
                    hit.append(name + "_plus1")
                if y == v - 1:
                    hit.append(name + "_minus1")
            if hit:
                prompt_ok_any = True
                for h in hit:
                    op_hits[op][h] += 1
        if prompt_ok_any:
            stats["prompts_with_any_supported_fit"] += 1
        else:
            stats["prompts_no_supported_fit"] += 1
            if len(samples) < args.show:
                samples.append({"id": str(r.get("id", "")), "n_eqs": str(len(eqs))})

    print("## Numeric Operator Rule Check")
    print(f"prompts_detected: {stats['prompts_detected']}")
    print(f"prompts_with_any_supported_fit: {stats['prompts_with_any_supported_fit']}")
    print(f"prompts_no_supported_fit: {stats['prompts_no_supported_fit']}")
    if stats["prompts_detected"]:
        print(
            "fit_rate:",
            f"{stats['prompts_with_any_supported_fit'] / stats['prompts_detected']:.3f}",
        )
    print("\n## operator -> top candidate families")
    for op, ctr in sorted(op_hits.items(), key=lambda kv: sum(kv[1].values()), reverse=True):
        print(op, ctr.most_common(8))
    if samples:
        print("\n## sample prompts with no supported fit")
        for s in samples:
            print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
