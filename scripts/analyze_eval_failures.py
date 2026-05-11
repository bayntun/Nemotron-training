"""
Summarize eval JSONL from tmp_train_csv_remote.py (eval_details.jsonl).
Groups failures by inferred answer type and prints counts + samples.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.I)
BINARY_RE = re.compile(r"^[01]+$")
INT_RE = re.compile(r"^-?[0-9]+$")
FLOAT_RE = re.compile(r"^-?[0-9]+\.[0-9]+$")


def answer_type(ans: str) -> str:
    s = ans.strip()
    if not s:
        return "empty"
    if BINARY_RE.match(s) and len(s) >= 4:
        return "binary"
    if ROMAN_RE.match(s):
        return "roman"
    if FLOAT_RE.match(s):
        return "float"
    if INT_RE.match(s):
        return "int"
    if " " in s:
        return "phrase"
    if re.search(r"[a-zA-Z]", s):
        return "word"
    return "special"


def _prompt_bucket(prompt: str) -> str:
    pl = prompt.lower()
    if "encrypt" in pl or "decrypt" in pl or "cipher" in pl:
        return "encrypt_or_cipher_text"
    if "binary" in pl or ("8-bit" in pl and "binary" in pl):
        return "binary_bits"
    if "roman" in pl or re.search(r"\b[ivxlcdm]{2,}\b", pl):
        return "roman_numeral"
    if "unit conversion" in pl:
        return "unit_conversion"
    if "gravitational" in pl or "falling distance" in pl:
        return "gravity_kinematics"
    if "->" in prompt or "→" in prompt:
        return "arrow_transform"
    if "equation" in pl or "transformation rules" in pl:
        return "equation_or_rule"
    return "other"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("eval_jsonl", type=Path, help="eval_details.jsonl path")
    p.add_argument("--metric", choices=["full", "first"], default="full", help="which ok_* field to use")
    p.add_argument(
        "--extra",
        action="store_true",
        help="Print prompt-bucket counts for failures and first-token-only successes.",
    )
    args = p.parse_args()
    ok_key = "ok_full" if args.metric == "full" else "ok_first"

    rows = [json.loads(line) for line in args.eval_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    correct = sum(1 for r in rows if r.get(ok_key))
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[answer_type(str(r.get("gt", "")))].append(r)

    print(f"file={args.eval_jsonl} n={total} accuracy_{args.metric}={correct}/{total}={correct/max(total,1):.4f}")
    print()

    for t in sorted(by_type.keys()):
        sub = by_type[t]
        c = sum(1 for r in sub if r.get(ok_key))
        print(f"## {t}  n={len(sub)} acc={c/len(sub):.3f}")
        fails = [r for r in sub if not r.get(ok_key)]
        # Top mistaken preds
        ctr = Counter(str(r.get("pred", "")) for r in fails[:500])
        print("   common wrong first-tokens:", ctr.most_common(8))
        for r in fails[:3]:
            print(
                "   -",
                json.dumps(
                    {
                        "id": r.get("id"),
                        "gt": r.get("gt"),
                        "pred": r.get("pred"),
                        "gen_preview": (r.get("gen_full") or "")[:120],
                    },
                    ensure_ascii=False,
                ),
            )
        print()

    if args.extra:
        fails_all = [r for r in rows if not r.get("ok_full")]
        print("## failure prompt buckets (ok_full=False)")
        bkt = Counter(_prompt_bucket(str(r.get("prompt", ""))) for r in fails_all)
        for k, v in bkt.most_common():
            print(f"   {k}: {v}")
        print()
        partial = [r for r in rows if r.get("ok_first") and not r.get("ok_full")]
        print(f"## first-token-only wins (ok_first & not ok_full): n={len(partial)}")
        bkt2 = Counter(_prompt_bucket(str(r.get("prompt", ""))) for r in partial)
        for k, v in bkt2.most_common(12):
            print(f"   {k}: {v}")
        print()
        # Length mismatch on phrase failures
        phrase_fails = [r for r in fails_all if answer_type(str(r.get("gt", ""))) == "phrase"]
        len_mis = 0
        for r in phrase_fails:
            g = (r.get("gen_full") or "").strip()
            gt = str(r.get("gt", "")).strip()
            if g and len(g) != len(gt):
                len_mis += 1
        if phrase_fails:
            print(
                f"## phrase failures where len(gen_full)!=len(gt): {len_mis}/{len(phrase_fails)} "
                "(character-length mismatch on full line)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
