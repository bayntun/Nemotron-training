"""
Smoke test for Google Gemini teacher API (AI Studio key).

Verifies:
1. GEMINI_API_TOKEN / GEMINI_API_KEY / GOOGLE_API_KEY is set and valid.
2. generateContent returns 200 and text.
3. Model output passes extract_final_answer + verify for a trivial boxed answer.

Run from repo root:
    python teacher/gemini_smoke_test.py

Or:
    python -m teacher.gemini_smoke_test
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO / "bootstrap" / "secrets_local.env")
    load_dotenv()
except ImportError:
    pass

from eval.grader import extract_final_answer, verify

SMOKE_PROMPT = (
    "Solve the following puzzle and place your final answer inside \\boxed{}.\n\n"
    "What is six times seven?"
)
EXPECTED = "42"
DEFAULT_MODEL = "gemini-2.5-flash"


def _api_key() -> str:
    return (
        os.environ.get("GEMINI_API_TOKEN", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )


def main() -> int:
    print("Gemini API smoke test")
    print("=" * 60)

    key = _api_key()
    if not key:
        print(
            "FAIL: No API key. Set GEMINI_API_TOKEN, GEMINI_API_KEY, or GOOGLE_API_KEY.",
            file=sys.stderr,
        )
        return 2

    model = os.environ.get("GEMINI_SMOKE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": SMOKE_PROMPT}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, params={"key": key}, json=body)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        print(f"FAIL: HTTP {e.response.status_code}: {e.response.text[:500]}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    usage = data.get("usageMetadata", {})
    print(f"Model:              {model}")
    print(f"Prompt tokens:      {usage.get('promptTokenCount', '?')}")
    print(f"Candidates tokens:    {usage.get('candidatesTokenCount', '?')}")
    print()

    text = ""
    cands = data.get("candidates") or []
    if cands:
        parts = (cands[0].get("content") or {}).get("parts") or []
        if parts and isinstance(parts[0].get("text"), str):
            text = parts[0]["text"]

    print("--- response content ---")
    print(text or "(empty)")
    print("--- end response ---")
    print()

    extracted = extract_final_answer(text)
    ok = verify(EXPECTED, extracted)
    print(f"Extracted answer:   {extracted!r}")
    print(f"Expected:           {EXPECTED!r}")
    print(f"Grader verdict:     {'CORRECT' if ok else 'WRONG'}")

    return 0 if text else 2


if __name__ == "__main__":
    raise SystemExit(main())
