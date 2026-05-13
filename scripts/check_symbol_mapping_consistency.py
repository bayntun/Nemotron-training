#!/usr/bin/env python3
"""
Check symbol->digit mapping consistency for 2-symbol arithmetic puzzles.

Assumed pattern per example line:
  XY + ZW = 123
where XY/ZW are 2-char symbol tokens and operator is one of +, -, *.

For each prompt, we:
1) extract example equations from lines
2) solve symbol->digit assignments (injective, cap by max symbols)
3) report whether mappings are unique / ambiguous / unsolved
4) aggregate cross-puzzle symbol conflicts
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

EQ_RE = re.compile(r"^\s*([^\s=]{2})\s*([+\-*])\s*([^\s=]{2})\s*=\s*(-?\d+)\s*$")


@dataclass(frozen=True)
class Eq:
    a1: str
    a2: str
    op: str
    b1: str
    b2: str
    y: int


def _extract_equations(prompt: str, symbol_only: bool) -> list[Eq]:
    out: list[Eq] = []
    for ln in prompt.splitlines():
        m = EQ_RE.match(ln)
        if not m:
            continue
        x, op, z, y = m.group(1), m.group(2), m.group(3), int(m.group(4))
        if symbol_only and (x.isdigit() or z.isdigit()):
            continue
        out.append(Eq(x[0], x[1], op, z[0], z[1], y))
    return out


def _eval_eq(eq: Eq, mp: dict[str, int]) -> bool:
    a = 10 * mp[eq.a1] + mp[eq.a2]
    b = 10 * mp[eq.b1] + mp[eq.b2]
    if eq.op == "+":
        return a + b == eq.y
    if eq.op == "-":
        return a - b == eq.y
    if eq.op == "*":
        return a * b == eq.y
    return False


def _solve(eqns: list[Eq], max_solutions: int = 50) -> list[dict[str, int]]:
    syms = sorted({s for e in eqns for s in (e.a1, e.a2, e.b1, e.b2)})
    freq = Counter([s for e in eqns for s in (e.a1, e.a2, e.b1, e.b2)])
    syms.sort(key=lambda s: (-freq[s], s))
    out: list[dict[str, int]] = []

    def backtrack(i: int, mp: dict[str, int], used: set[int]) -> None:
        if len(out) >= max_solutions:
            return
        if i == len(syms):
            if all(_eval_eq(e, mp) for e in eqns):
                out.append(dict(mp))
            return
        s = syms[i]
        for d in range(10):
            if d in used:
                continue
            mp[s] = d
            used.add(d)
            ok = True
            for e in eqns:
                needed = (e.a1, e.a2, e.b1, e.b2)
                if all(k in mp for k in needed):
                    if not _eval_eq(e, mp):
                        ok = False
                        break
            if ok:
                backtrack(i + 1, mp, used)
            used.remove(d)
            del mp[s]

    backtrack(0, {}, set())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path, help="CSV with prompt column")
    ap.add_argument("--prompt-col", default="prompt")
    ap.add_argument("--min-eqns", type=int, default=2)
    ap.add_argument("--max-symbols", type=int, default=7)
    ap.add_argument("--max-solutions", type=int, default=50)
    ap.add_argument("--symbol-only", action="store_true", help="Only keep examples where both operands are non-digit symbol pairs.")
    ap.add_argument("--show", type=int, default=8, help="sample rows per bucket")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open("r", encoding="utf-8")))
    stats = Counter()
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    global_fixed: dict[str, Counter[int]] = defaultdict(Counter)

    for r in rows:
        p = str(r.get(args.prompt_col, ""))
        eqns = _extract_equations(p, symbol_only=args.symbol_only)
        if len(eqns) < args.min_eqns:
            continue
        stats["detected_prompts"] += 1
        syms = sorted({s for e in eqns for s in (e.a1, e.a2, e.b1, e.b2)})
        if len(syms) > args.max_symbols:
            stats["skipped_too_many_symbols"] += 1
            continue
        sols = _solve(eqns, max_solutions=args.max_solutions)
        if not sols:
            stats["unsolved"] += 1
            if len(samples["unsolved"]) < args.show:
                samples["unsolved"].append({"id": str(r.get("id", "")), "eqns": str(len(eqns)), "syms": "".join(syms)})
            continue
        if len(sols) == 1:
            stats["unique_mapping"] += 1
        elif len(sols) >= args.max_solutions:
            stats["many_mappings_capped"] += 1
        else:
            stats["multiple_mappings"] += 1

        fixed_here = 0
        for s in syms:
            vals = {m[s] for m in sols}
            if len(vals) == 1:
                fixed_here += 1
                v = next(iter(vals))
                global_fixed[s][v] += 1
        stats["symbols_total_in_solved"] += len(syms)
        stats["symbols_fixed_in_solved"] += fixed_here

        bucket = "solved_unique" if len(sols) == 1 else "solved_multi"
        if len(samples[bucket]) < args.show:
            samples[bucket].append({"id": str(r.get("id", "")), "eqns": str(len(eqns)), "syms": "".join(syms), "n_solutions": str(len(sols))})

    conflict_symbols = 0
    for s, ctr in global_fixed.items():
        if len(ctr) > 1:
            conflict_symbols += 1
    stats["global_symbol_conflicts"] = conflict_symbols
    stats["global_symbols_fixed_once_or_more"] = len(global_fixed)

    print("## Symbol Mapping Consistency Report")
    for k in (
        "detected_prompts",
        "unique_mapping",
        "multiple_mappings",
        "many_mappings_capped",
        "unsolved",
        "skipped_too_many_symbols",
        "symbols_total_in_solved",
        "symbols_fixed_in_solved",
        "global_symbols_fixed_once_or_more",
        "global_symbol_conflicts",
    ):
        print(f"{k}: {stats.get(k, 0)}")
    if stats.get("symbols_total_in_solved", 0):
        r = stats["symbols_fixed_in_solved"] / stats["symbols_total_in_solved"]
        print(f"fixed_symbol_rate_in_solved: {r:.3f}")

    for bucket in ("solved_unique", "solved_multi", "unsolved"):
        if not samples[bucket]:
            continue
        print(f"\n## samples_{bucket}")
        for s in samples[bucket]:
            print(s)

    if global_fixed:
        print("\n## global_symbol_digit_votes (top)")
        items = sorted(global_fixed.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:20]
        for sym, ctr in items:
            print(sym, dict(ctr.most_common(5)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
