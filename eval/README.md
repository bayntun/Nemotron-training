# `eval/`

Local evaluation harness that mirrors the Kaggle scoring kernel exactly.

## What lives here

- **`grader.py`** — a verbatim port of the competition's `extract_final_answer`
  and `verify` functions. **Do not modify.** If the leaderboard grader changes,
  re-port from the kernel; do not patch in place.
- **`test_grader.py`** — pins down the grader's behavior across boxed
  extraction, fallback patterns, numeric tolerance (`rel_tol=1e-2,
  abs_tol=1e-5`), and case-insensitive string equality.
- **`greedy_harness.py`** — vLLM-based eval that loads the Nemotron base +
  optional LoRA adapter and decodes greedily with the kernel-identical
  parameters. Runs on the V100 server or Kaggle Blackwell, not on Windows.

## Kernel parameters (locked)

```python
KERNEL_PARAMS = {
    "max_lora_rank": 32,
    "max_model_len": 8192,
    "max_num_seqs": 64,
    "gpu_memory_utilization": 0.85,
}
KERNEL_SAMPLING = {
    "temperature": 0.0,   # greedy decode -- NO test-time sampling
    "top_p": 1.0,
    "max_tokens": 7680,
}
```

## Running tests (Windows, no GPU needed)

```powershell
python -m pytest eval/test_grader.py -v
```

## Running greedy eval (V100 server / Kaggle Blackwell)

```bash
python -m eval.greedy_harness \
    --base-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --adapter ./adapters/sft_baseline \
    --val-jsonl ./data/cache/sft/validation.jsonl \
    --out ./outputs/eval_sft_baseline.jsonl \
    --dtype float16
```

Use ``--dtype bfloat16`` on hardware that matches the competition kernel; Volta V100 often needs ``float16``.
