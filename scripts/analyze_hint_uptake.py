"""Measure how often each hint substring appears in prompt_augmented for passes vs fails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MARKERS = [
    ("numeric", "Numerical baseline"),
    ("cipher_length", "Structural hint: each listed example"),
    ("encrypt_lexical", "Decrypt / cipher lexical"),
    ("equation_shape", "Equation / operator puzzle"),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("eval_jsonl", type=Path)
    args = p.parse_args()
    rows = [json.loads(l) for l in args.eval_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    fails = [r for r in rows if not r.get("ok_full")]
    oks = [r for r in rows if r.get("ok_full")]
    n = len(rows)
    print(f"file={args.eval_jsonl} n={n} acc_full={len(oks)/n:.4f} fails={len(fails)}")
    print("\n## Among failures (ok_full=False): rows whose prompt_augmented contains:")
    pa_key = "prompt_augmented"
    for lab, sub in MARKERS:
        c = sum(1 for r in fails if sub in (r.get(pa_key) or ""))
        print(f"   {lab:16} {c:4} / {len(fails)}  ({c/max(len(fails),1):.1%})")
    none = sum(
        1
        for r in fails
        if not any(sub in (r.get(pa_key) or "") for _, sub in MARKERS)
    )
    print(f"   {'none_of_four':16} {none:4} / {len(fails)}  ({none/max(len(fails),1):.1%})")
    missing_key = sum(1 for r in fails if pa_key not in r)
    if missing_key:
        print(f"\n   (rows missing '{pa_key}' field: {missing_key})")
    print("\n## Among successes (for contrast):")
    for lab, sub in MARKERS:
        c = sum(1 for r in oks if sub in (r.get(pa_key) or ""))
        print(f"   {lab:16} {c:4} / {len(oks)}  ({c/max(len(oks),1):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
