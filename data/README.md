# `data/`

Holds dataset download scripts and (gitignored) cached dataset files.

## What lives here

- `download.py` — fetches the andy279 datasets from HuggingFace.
- `cache/` (gitignored) — local download destination.
  - `cache/sft/` — cleaned SFT data, ~408 MB, 49,290 examples / 7,200 puzzles.
  - `cache/traces/` — raw teacher traces, ~1.02 GB, includes failed attempts and full metadata.

## One-time setup

1. HuggingFace account + token (read scope) at <https://huggingface.co/settings/tokens>.
2. Set `HF_TOKEN` in your repo-local `.env` file (copy `.env.example` to `.env`).
3. Visit each of these dataset pages and click **"Agree and access repository"**:
   - <https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge>
   - <https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces>

   Both repos are gated; the click-through ToS must be accepted on each before
   any token can pull files.

## Running

```bash
python -m data.download                # downloads both
python -m data.download --sft-only     # cleaned SFT only (~408 MB)
python -m data.download --traces-only  # raw traces only (~1.02 GB)
```

## Schema (raw traces)

Each line of `cache/traces/all_traces_merged.jsonl` is one puzzle:

```json
{
  "id": "001b24c4",
  "prompt": "In Alice's Wonderland, ...",
  "ground_truth": "XXXVIII",
  "attempts": [
    {
      "attempt_idx": 0,
      "reasoning": "",
      "content": "The answer is XXXVIII",
      "predicted_answer": "XXXVIII",
      "is_correct": true,
      "is_correct_official": true,
      "temperature": 1.0,
      "usage": {"prompt_tokens": 123, "completion_tokens": 456}
    }
  ],
  "correct_count": 3,
  "correct_count_official": 3,
  "total_attempts": 4
}
```

Note: `is_correct` uses the fixed grader; `is_correct_official` reproduces the
Kaggle grader's known binary-string bug. **Use `is_correct_official` when
re-filtering**, since that's what the leaderboard scores against.
