#!/usr/bin/env python3
"""
Stage 2: DeepSeek teacher traces on eval failures, keep only grader-verified rows.

Reads ``eval_details.jsonl`` from ``tmp_train_csv_remote.py`` (one JSON object per line).
Filters to ``ok_full == false``, optionally by prompt bucket (same buckets as
``scripts/analyze_eval_failures.py``).

Modes:
  challenge (default): model must solve; row kept only if ``grade(response, gt)``.
  strict: model is told ``gt`` and must end with ``\\boxed{gt}``; still verified.

Outputs:
  - JSONL: full teacher response + metadata for audit.
  - Optional CSV: ``id,prompt,answer`` (raw prompt + ground truth) ready to merge into train.csv.

Run (from repo root, with DEEPSEEK_API_KEY in env or .env):

  python scripts/stage2_deepseek_verified_synth.py \\
    path/to/eval_details.jsonl \\
    --out-jsonl data/cache/synth_verified.jsonl \\
    --out-csv data/cache/synth_verified.csv \\
    --bucket equation_or_rule --limit 30

Dry-run (no API):

  python scripts/stage2_deepseek_verified_synth.py eval.jsonl --dry-run --limit 5

Progress (long runs): by default, ``--progress`` is enabled when 20+ rows are selected
(``progress N/T verified=V`` on stdout every few completions). Override with ``--no-progress``.
Output files are still written only after all API calls finish.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path

# Repo root on sys.path when running as ``python scripts/...``
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.grader import grade  # noqa: E402
from teacher.deepseek_client import DeepSeekClient  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO / "bootstrap" / "secrets_local.env")
    load_dotenv()
except ImportError:
    pass

BUCKETS = (
    "all",
    "encrypt_or_cipher_text",
    "binary_bits",
    "roman_numeral",
    "unit_conversion",
    "gravity_kinematics",
    "arrow_transform",
    "equation_or_rule",
    "other",
)


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


SYS_CHALLENGE = (
    "You solve Nemotron-style Wonderland puzzles. Think briefly, then give the final "
    "answer inside \\boxed{...} only. The boxed string must match the puzzle's required "
    "format (spacing, digits, case as appropriate)."
)

SYS_STRICT = (
    "You write a short chain-of-thought for a training example. You will be given the "
    "correct final answer; your job is to justify it from the prompt's examples, then "
    "output that exact answer inside \\boxed{...}."
)


def _user_challenge(prompt_text: str) -> str:
    return f"Puzzle:\n\n{prompt_text.strip()}\n\nSolve it and end with \\boxed{{your answer}}."


def _user_strict(prompt_text: str, gt: str) -> str:
    return (
        f"Puzzle:\n\n{prompt_text.strip()}\n\n"
        f"The correct final answer is: {gt}\n"
        "Explain briefly why this follows from the examples, then output exactly:\n"
        f"\\boxed{{{gt}}}\n"
        "Do not change the boxed content."
    )


async def _one(
    client: DeepSeekClient,
    sem: asyncio.Semaphore,
    row: dict,
    *,
    mode: str,
    use_augmented: bool,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict | None:
    raw = (row.get("prompt") or "").strip()
    aug = (row.get("prompt_augmented") or "").strip()
    text = aug if use_augmented and aug else raw
    if not text:
        return None
    gt = str(row.get("gt", "")).strip()
    if not gt:
        return None

    if mode == "strict":
        messages = [
            {"role": "system", "content": SYS_STRICT},
            {"role": "user", "content": _user_strict(text, gt)},
        ]
    else:
        messages = [
            {"role": "system", "content": SYS_CHALLENGE},
            {"role": "user", "content": _user_challenge(text)},
        ]

    async with sem:
        resp = await client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    ok = grade(resp.content, gt)
    out = {
        "id": row.get("id"),
        "prompt": raw,
        "answer": gt,
        "mode": mode,
        "teacher_model": resp.model,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "verified": ok,
        "teacher_text": resp.content,
    }
    return out if ok else None


async def _run_all(
    rows: list[dict],
    *,
    mode: str,
    use_augmented: bool,
    model: str,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    progress: bool,
) -> list[dict]:
    total = len(rows)
    if total == 0:
        return []

    sem = asyncio.Semaphore(concurrency)

    async with DeepSeekClient() as client:

        async def _indexed(i: int, r: dict) -> tuple[int, dict | None]:
            rec = await _one(
                client,
                sem,
                r,
                mode=mode,
                use_augmented=use_augmented,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return i, rec

        tasks = [asyncio.create_task(_indexed(i, r)) for i, r in enumerate(rows)]
        by_idx: dict[int, dict | None] = {}
        n_done = 0
        n_verified = 0
        step = max(1, min(50, total // 20))
        for fut in asyncio.as_completed(tasks):
            i, rec = await fut
            by_idx[i] = rec
            n_done += 1
            if rec is not None:
                n_verified += 1
            if progress and (n_done % step == 0 or n_done == total):
                pct = 100.0 * n_done / total
                print(
                    f"progress {n_done}/{total} verified={n_verified} ({pct:.0f}%)",
                    flush=True,
                )

    return [by_idx[i] for i in range(total) if by_idx[i] is not None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_jsonl", type=Path, help="eval_details.jsonl from tmp_train_csv_remote.py")
    ap.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Write verified rows as JSONL (required unless --dry-run).",
    )
    ap.add_argument("--out-csv", type=Path, default=None, help="Optional id,prompt,answer for train.csv merge")
    ap.add_argument("--bucket", choices=BUCKETS, default="equation_or_rule")
    ap.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max rows to send to API after filtering (-1 = all matching rows).",
    )
    ap.add_argument("--mode", choices=("challenge", "strict"), default="challenge")
    ap.add_argument("--use-augmented-prompt", action="store_true", help="Show model augmented User prompt (hints).")
    ap.add_argument("--model", type=str, default="deepseek-chat")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print progress during API calls (default: on when 20+ rows are selected).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print selection only; no API calls")
    args = ap.parse_args()

    lines = [ln for ln in args.eval_jsonl.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    rows_in = [json.loads(ln) for ln in lines]

    candidates: list[dict] = []
    for row in rows_in:
        if row.get("ok_full"):
            continue
        p = str(row.get("prompt", ""))
        b = _prompt_bucket(p)
        if args.bucket != "all" and b != args.bucket:
            continue
        candidates.append(row)

    if args.limit >= 0:
        candidates = candidates[: args.limit]
    print(f"selected={len(candidates)} bucket={args.bucket} mode={args.mode}", flush=True)

    if args.dry_run:
        for r in candidates[:5]:
            print(json.dumps({"id": r.get("id"), "bucket": _prompt_bucket(str(r.get("prompt", "")))}, ensure_ascii=False))
        if len(candidates) > 5:
            print(f"... and {len(candidates) - 5} more", flush=True)
        return 0

    if not args.out_jsonl:
        print("ERROR: --out-jsonl is required when not using --dry-run", file=sys.stderr)
        return 2

    if args.progress is None:
        args.progress = len(candidates) >= 20

    kept = asyncio.run(
        _run_all(
            candidates,
            mode=args.mode,
            use_augmented=args.use_augmented_prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            progress=args.progress,
        )
    )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as w:
        for rec in kept:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"verified_rows={len(kept)} written_jsonl={args.out_jsonl}", flush=True)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["id", "prompt", "answer"])
            wr.writeheader()
            for rec in kept:
                wr.writerow({"id": rec.get("id", ""), "prompt": rec["prompt"], "answer": rec["answer"]})
        print(f"wrote_csv={args.out_csv}", flush=True)

    if not kept:
        print("WARN: zero verified rows (try --mode strict, --bucket all, or higher --limit)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
