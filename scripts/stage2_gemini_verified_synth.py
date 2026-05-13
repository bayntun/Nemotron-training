#!/usr/bin/env python3
"""
Stage 2: Gemini teacher traces on eval failures, keep only grader-verified rows.

Same selection / buckets / modes as ``stage2_deepseek_verified_synth.py``.
Reads API key from ``GEMINI_API_TOKEN`` or ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``.

Example::

  python scripts/stage2_gemini_verified_synth.py eval.jsonl \\
    --out-jsonl data/cache/synth_gemini.jsonl --out-csv data/cache/synth_gemini.csv \\
    --bucket equation_or_rule --limit 20 --model gemini-2.5-flash \\
    --target-rpm 140 --verbose-429

Dry-run::

  python scripts/stage2_gemini_verified_synth.py eval.jsonl --dry-run --limit 5
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import httpx

from eval.grader import grade  # noqa: E402

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


class _AsyncRateGate:
    """At least ``min_interval`` seconds between successive HTTP attempts (request starts)."""

    def __init__(self, min_interval: float) -> None:
        self._min = max(0.0, float(min_interval))
        self._lock = asyncio.Lock()
        self._last_start: float | None = None

    async def acquire(self) -> None:
        if self._min <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            if self._last_start is not None:
                wait = self._min - (now - self._last_start)
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last_start = time.monotonic()


def _gemini_error_message(resp: httpx.Response) -> str:
    """Short error text from a Gemini JSON error body (or raw prefix)."""
    try:
        payload = resp.json()
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict):
            parts = [str(x) for x in (err.get("status"), err.get("message")) if x]
            return " | ".join(parts)[:800]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return (resp.text or "")[:500]


def _gemini_429_log_line(resp: httpx.Response, *, attempt: int, max_attempts: int, wait_s: float) -> str:
    ra = resp.headers.get("Retry-After", "")
    hint = _gemini_error_message(resp)
    return (
        f"gemini 429 attempt {attempt + 1}/{max_attempts} "
        f"retry_after_header={ra!r} sleep_s={wait_s:.1f} detail={hint!r}"
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


def _gemini_key() -> str:
    return (
        os.environ.get("GEMINI_API_TOKEN", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
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


async def _gemini_generate(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_output_tokens: int,
    rate_gate: _AsyncRateGate | None,
    verbose_429: bool,
) -> tuple[str, int, int]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body: dict = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    data: dict = {}
    last_resp: httpx.Response | None = None
    max_attempts = 15
    for attempt in range(max_attempts):
        if rate_gate is not None:
            await rate_gate.acquire()
        resp = await client.post(
            url,
            params={"key": api_key},
            json=body,
            timeout=120.0,
        )
        last_resp = resp
        if resp.status_code == 200:
            data = resp.json()
            break
        if resp.status_code == 429:
            msg = _gemini_error_message(resp)
            ml = msg.lower()
            if "spending cap" in ml or "ai.studio/spend" in ml:
                print(_gemini_429_log_line(resp, attempt=attempt, max_attempts=max_attempts, wait_s=0.0), file=sys.stderr, flush=True)
                raise RuntimeError(
                    "Gemini HTTP 429: this Google AI project exceeded its monthly spending cap "
                    "(RESOURCE_EXHAUSTED). Retries will not help. Raise or remove the cap in "
                    "AI Studio (Spend / billing) or use a different API key/project. "
                    "https://ai.google.dev/gemini-api/docs/billing#project-spend-caps"
                ) from None
            ra = resp.headers.get("Retry-After")
            if ra and ra.replace(".", "", 1).isdigit():
                wait_s = min(120.0, float(ra))
            else:
                wait_s = min(90.0, 2.0 * (2 ** min(attempt, 6)))
            print(_gemini_429_log_line(resp, attempt=attempt, max_attempts=max_attempts, wait_s=wait_s), file=sys.stderr, flush=True)
            if verbose_429:
                raw = (resp.text or "")[:800].replace(api_key, "<redacted>")
                print(f"gemini 429 raw_body_prefix={raw!r}", file=sys.stderr, flush=True)
            await asyncio.sleep(wait_s)
            continue
        if resp.status_code in (500, 502, 503, 504):
            await asyncio.sleep(min(30.0, 2.0 * (attempt + 1)))
            continue
        resp.raise_for_status()
    else:
        if last_resp is not None and last_resp.status_code == 429:
            raise RuntimeError(
                "Gemini API rate limited (HTTP 429) after 15 attempts. "
                "Console RPM/TPM can still allow 429 when RPD is exhausted, TPM is exceeded on "
                "large prompts, or the API key belongs to a different project than the quota you "
                "are viewing. Retry later, use --target-rpm below your RPM cap, and run with "
                "--verbose-429 once to print Google's error detail (stderr only)."
            ) from None
        if last_resp is not None:
            last_resp.raise_for_status()
        raise RuntimeError("Gemini: no response after retries")

    usage = data.get("usageMetadata", {})
    prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
    completion_tokens = int(usage.get("candidatesTokenCount", 0) or 0)

    text_out = ""
    cands = data.get("candidates") or []
    if cands:
        parts = (cands[0].get("content") or {}).get("parts") or []
        if parts and isinstance(parts[0].get("text"), str):
            text_out = parts[0]["text"]
    return text_out, prompt_tokens, completion_tokens


async def _one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    model: str,
    row: dict,
    *,
    mode: str,
    use_augmented: bool,
    temperature: float,
    max_output_tokens: int,
    rate_gate: _AsyncRateGate | None,
    verbose_429: bool,
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
        system, user = SYS_STRICT, _user_strict(text, gt)
    else:
        system, user = SYS_CHALLENGE, _user_challenge(text)

    async with sem:
        content, pt, ct = await _gemini_generate(
            client,
            api_key=api_key,
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            rate_gate=rate_gate,
            verbose_429=verbose_429,
        )
    if not grade(content, gt):
        return None
    return {
        "id": row.get("id"),
        "prompt": raw,
        "answer": gt,
        "mode": mode,
        "teacher_model": model,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "verified": True,
        "teacher_text": content,
    }


async def _run_all(
    rows: list[dict],
    *,
    api_key: str,
    model: str,
    mode: str,
    use_augmented: bool,
    temperature: float,
    max_output_tokens: int,
    concurrency: int,
    target_rpm: float,
    verbose_429: bool,
) -> list[dict]:
    min_interval = 0.0
    if target_rpm and target_rpm > 0:
        min_interval = max(0.05, (60.0 / float(target_rpm)) * 1.05)
    rate_gate = _AsyncRateGate(min_interval) if min_interval > 0 else None
    if rate_gate is not None:
        print(f"rate_gate min_interval_s={min_interval:.3f} (target_rpm={target_rpm})", flush=True)

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        if concurrency <= 1:
            out: list[dict | None] = []
            for r in rows:
                out.append(
                    await _one(
                        client,
                        sem,
                        api_key,
                        model,
                        r,
                        mode=mode,
                        use_augmented=use_augmented,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        rate_gate=rate_gate,
                        verbose_429=verbose_429,
                    )
                )
                await asyncio.sleep(2.0)
            return [x for x in out if x is not None]
        tasks = [
            _one(
                client,
                sem,
                api_key,
                model,
                r,
                mode=mode,
                use_augmented=use_augmented,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                rate_gate=rate_gate,
                verbose_429=verbose_429,
            )
            for r in rows
        ]
        results = await asyncio.gather(*tasks)
    return [x for x in results if x is not None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_jsonl", type=Path)
    ap.add_argument("--out-jsonl", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--bucket", choices=BUCKETS, default="equation_or_rule")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--mode", choices=("challenge", "strict"), default="challenge")
    ap.add_argument("--use-augmented-prompt", action="store_true")
    ap.add_argument("--model", type=str, default="gemini-2.5-flash")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument(
        "--target-rpm",
        type=float,
        default=0.0,
        help=(
            "If >0, enforce at least 60/target_rpm seconds between each HTTP attempt "
            "(per-process). Use e.g. 140 when your Gemini console shows ~150 RPM."
        ),
    )
    ap.add_argument(
        "--verbose-429",
        action="store_true",
        help="On HTTP 429, also print a redacted snippet of the response body to stderr.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = _gemini_key()
    if not args.dry_run and not key:
        print(
            "ERROR: Set GEMINI_API_TOKEN, GEMINI_API_KEY, or GOOGLE_API_KEY (or add to bootstrap/secrets_local.env).",
            file=sys.stderr,
        )
        return 2

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

    candidates = candidates[: args.limit]
    print(f"selected={len(candidates)} bucket={args.bucket} mode={args.mode} model={args.model}", flush=True)

    if args.dry_run:
        for r in candidates[:5]:
            print(json.dumps({"id": r.get("id"), "bucket": _prompt_bucket(str(r.get("prompt", "")))}, ensure_ascii=False))
        if len(candidates) > 5:
            print(f"... and {len(candidates) - 5} more", flush=True)
        return 0

    if not args.out_jsonl:
        print("ERROR: --out-jsonl required without --dry-run", file=sys.stderr)
        return 2

    kept = asyncio.run(
        _run_all(
            candidates,
            api_key=key,
            model=args.model,
            mode=args.mode,
            use_augmented=args.use_augmented_prompt,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            concurrency=args.concurrency,
            target_rpm=args.target_rpm,
            verbose_429=args.verbose_429,
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
        print("WARN: zero verified rows (try --mode strict or different --model)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
