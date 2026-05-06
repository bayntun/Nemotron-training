# `teacher/`

DeepSeek V3.2 teacher CoT generation for Phase 2.

## What lives here

- **`deepseek_client.py`** — async OpenAI-compatible client for DeepSeek V3.2,
  with tenacity retry/backoff. Compatible with OpenRouter (set
  `DEEPSEEK_BASE_URL=https://openrouter.ai/api/v1`) as a drop-in fallback.
- **`smoke_test.py`** — ~$0.001 round-trip that validates the API key, the
  endpoint, and that our local grader's `extract_final_answer` parses the
  model's output correctly. Run during Phase 0.

## One-time setup

1. Account + payment method at <https://platform.deepseek.com>. Load $20 to
   start; with the off-peak discount this likely covers all of Phase 2.
2. Generate an API key at <https://platform.deepseek.com/api_keys>.
3. Add `DEEPSEEK_API_KEY` to `.env` (copy `.env.example` to `.env`).

## Running the smoke test

```powershell
python -m teacher.smoke_test
```

Expected output: a non-empty completion, a `\boxed{42}` answer, and
`Grader verdict: CORRECT`. If the verdict is `WRONG` but the completion is
non-empty, the infrastructure is fine -- the model just chose a slightly
non-standard answer format. Inspect the printed response.

## Cost / pricing notes

- `deepseek-chat` (V3.2-Exp): ~$0.14/M input, ~$0.28/M output (peak).
- Off-peak discount: 50% off between **16:30 - 00:30 UTC**. Schedule large
  batch jobs during this window.
- Realistic Phase 2 budget: ~45 M tokens total -> **$15-25** at peak rates.
- Rate limit fallbacks: shift to off-peak, or set `DEEPSEEK_BASE_URL` to
  OpenRouter's proxy.

## Models

| Model | Use case |
| --- | --- |
| `deepseek-chat` | Default for solver-guided CoT (Phase 2 bulk). Non-thinking mode by default; CoT comes from the prompt. |
| `deepseek-reasoner` | R1-style internal CoT. Use only for the hardest residual transformation puzzles. Slower and costlier. |
