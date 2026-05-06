"""
Phase 0 smoke test for DeepSeek V3.2 teacher generation.

Verifies that:
1. DEEPSEEK_API_KEY is set and the key is valid.
2. The /v1/chat/completions endpoint is reachable.
3. The model produces output ending in \\boxed{...}.
4. Our local grader's extract_final_answer + verify pipeline accepts the output.

Cost: a few hundred tokens, fractions of a US cent.

Run:
    python -m teacher.smoke_test
"""

from __future__ import annotations

import asyncio
import sys

from eval.grader import extract_final_answer, verify
from teacher.deepseek_client import DeepSeekClient

# A trivially solvable puzzle in the same style as the competition prompts.
# Expected boxed answer is "42".
SMOKE_PROMPT = (
    "Solve the following puzzle and place your final answer inside \\boxed{}.\n\n"
    "What is six times seven?"
)
EXPECTED = "42"


async def main() -> int:
    print("DeepSeek V3.2 smoke test")
    print("=" * 60)

    async with DeepSeekClient() as client:
        try:
            resp = await client.chat(
                messages=[{"role": "user", "content": SMOKE_PROMPT}],
                model="deepseek-chat",
                temperature=0.0,
                max_tokens=512,
            )
        except Exception as e:
            print(f"FAIL: API call raised {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    print(f"Model:             {resp.model}")
    print(f"Finish reason:     {resp.finish_reason}")
    print(f"Prompt tokens:     {resp.prompt_tokens}")
    print(f"Completion tokens: {resp.completion_tokens}")
    print()
    print("--- response content ---")
    print(resp.content)
    print("--- end response ---")
    print()

    extracted = extract_final_answer(resp.content)
    is_correct = verify(EXPECTED, extracted)

    print(f"Extracted answer:  {extracted!r}")
    print(f"Expected:          {EXPECTED!r}")
    print(f"Grader verdict:    {'CORRECT' if is_correct else 'WRONG'}")

    if not is_correct:
        print(
            "\nNOTE: 'WRONG' here likely means the model didn't use \\boxed{} or wrote\n"
            "      something like '42.0' that still parses correctly. Inspect the\n"
            "      response above. The infrastructure is fine if we got a 200 OK\n"
            "      and non-empty content.",
            file=sys.stderr,
        )

    return 0 if (resp.finish_reason and resp.content) else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
