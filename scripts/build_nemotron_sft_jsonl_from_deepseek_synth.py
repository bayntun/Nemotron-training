#!/usr/bin/env python3
"""
Build a HuggingFace-friendly JSONL for ``train/sft.py`` from DeepSeek stage2 JSONL.

Each output line has ``prompt`` + ``answer`` where ``answer`` is ``teacher_text``
(grader-verified CoT ending in \\boxed{...}). ``train/_dataset.py`` maps that to
messages for Nemotron chat template + SFTTrainer.

Optional ``--holdout-fraction`` (with ``--seed``) shuffles eligible rows, writes the
majority to ``train.jsonl`` and the rest to ``val_greedy.jsonl`` (``id``, ``prompt``,
``ground_truth`` from boxed teacher, ``category``) for ``eval.greedy_harness``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _import_extract_final_answer():
    sys.path.insert(0, str(_REPO))
    from eval.grader import extract_final_answer

    return extract_final_answer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-jsonl",
        type=Path,
        default=_REPO / "data" / "cache" / "synth_deepseek_full.jsonl",
        help="Stage2 DeepSeek JSONL (with teacher_text).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "data" / "cache" / "nemotron_sft_deepseek",
        help="Directory to write train.jsonl into.",
    )
    ap.add_argument("--limit", type=int, default=0, help="If >0, cap rows written to train.jsonl.")
    ap.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help=(
            "If in (0,1), reserve the last fraction of **eligible** rows for val_greedy.jsonl "
            "(competition-style fields) and write the leading rows to train.jsonl only. "
            "Use with the same --seed for reproducibility."
        ),
    )
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed when holdout-fraction > 0.")
    args = ap.parse_args()

    if not args.in_jsonl.is_file():
        print(f"ERROR: missing {args.in_jsonl}", file=sys.stderr)
        return 2

    if not (0.0 <= args.holdout_fraction < 1.0):
        print("ERROR: --holdout-fraction must be in [0, 1).", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val_greedy.jsonl"

    rows: list[tuple[str, str, str]] = []
    for line in args.in_jsonl.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        prompt = str(row.get("prompt") or "").strip()
        teacher = str(row.get("teacher_text") or "").strip()
        rid = str(row.get("id") or "").strip()
        if not prompt or not teacher:
            continue
        if not rid:
            rid = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        rows.append((rid, prompt, teacher))

    if args.holdout_fraction > 0.0:
        import random

        rng = random.Random(args.seed)
        rng.shuffle(rows)
        n_hold = int(round(len(rows) * args.holdout_fraction))
        n_hold = max(1, min(n_hold, len(rows) - 1)) if len(rows) > 1 else 0
        val_rows = rows[-n_hold:] if n_hold else []
        train_rows = rows[:-n_hold] if n_hold else list(rows)
    else:
        train_rows = list(rows)
        val_rows = []

    if args.limit and train_rows:
        train_rows = train_rows[: args.limit]

    with out_path.open("w", encoding="utf-8") as fout:
        for _rid, prompt, teacher in train_rows:
            fout.write(json.dumps({"prompt": prompt, "answer": teacher}, ensure_ascii=False) + "\n")

    n_val = 0
    if val_rows:
        extract_final_answer = _import_extract_final_answer()
        with val_path.open("w", encoding="utf-8") as vout:
            for rid, prompt, teacher in val_rows:
                gt = extract_final_answer(teacher)
                if not gt or gt == "NOT_FOUND":
                    continue
                vout.write(
                    json.dumps(
                        {
                            "id": rid,
                            "prompt": prompt,
                            "ground_truth": gt,
                            "category": "deepseek_holdout",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_val += 1

    print(f"wrote {len(train_rows)} rows -> {out_path}", flush=True)
    if val_rows:
        print(f"wrote {n_val} rows -> {val_path} (holdout fraction={args.holdout_fraction}, seed={args.seed})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
