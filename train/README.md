# `train/`

Training scripts for SFT (Phase 1), high-rank teacher + KD (Phase 3), and
optionally GRPO (Phase 4, deprioritized).

## Phase 1 — `sft.py` (implemented)

QLoRA + TRL **`SFTTrainer`** on the competition base model
(`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`). **Linux + CUDA only**
(JupyterHub / V100 server).

**Data:** run `python -m data.download --sft-only` first, or pass
`--dataset-name andy279/nemotron-reasoning-challenge` with `HF_TOKEN`.

**Dry-run** (no GPU; checks tokenizer + dataset columns):

```bash
python -m train.sft --output-dir ./outputs/_dry --dry-run --max-samples 4
```

**Single GPU** (example):

```bash
python -m train.sft --output-dir ./outputs/sft_baseline
```

**4× GPU + DeepSpeed ZeRO-2** (tune `num_processes` in the YAML):

```bash
accelerate launch --config_file configs/accelerate_zero2.yaml -m train.sft \
  --output-dir ./outputs/sft_baseline
```

**Packaging:** after training, point `python -m submit.package --adapter <output-dir>` at the saved adapter directory (once weights exist).

### Flags worth knowing

| Flag | Purpose |
|------|--------|
| `--peft-simple-targets` | Use `in_proj`/`out_proj` + `layers_to_transform` instead of regex targets (if regex matches nothing). |
| `--lora-attn-fallback` | LoRA only on `q_proj`/`k_proj`/`v_proj`/`o_proj` — not competition-canonical mixer targets, but useful to validate the Nemotron + TRL + 4bit stack when mixer LoRA hits shape errors. |
| `--no-thinking-template` | Omit `enable_thinking` in chat-template kwargs. |
| `--max-samples N` | Smoke subset. |
| `--attn-implementation eager` | If `sdpa` misbehaves on your stack. |
| `--nemotron-mamba-fused-kernels` | Opt into fused Mamba CUDA kernels (faster). **Off by default under 4bit**, which sets `is_fast_path_available=False` in `modeling_nemotron_h` so the `torch_forward` path runs (fused kernels pass raw `out_proj.weight` into `F.linear` and break BitsAndBytes). |

**Nemotron-H:** keep **`gradient_checkpointing=False`** (unsupported / hangs). TRL’s `SFTConfig` turns this off explicitly.

Under **4bit**, this trainer disables fused Mamba kernels by default (`is_fast_path_available=False`); the **`torch_forward`** path is correct with BitsAndBytes but uses **more activation memory** than fused kernels. If backward OOMs on a small shard (e.g. 16GB), lower **`--max-length`** (the detached launcher uses a short cap for smoke).

### Nemotron + DeepSeek teacher JSONL (remote)

- Build dataset: `python scripts/build_nemotron_sft_jsonl_from_deepseek_synth.py` → `data/cache/nemotron_sft_deepseek/train.jsonl`.
- Push + sync scripts as needed (`scripts/push_nemotron_sft_dataset_remote.py`, `scripts/_sync_nemotron_sft_to_remote.py`).
- **Detached launch** (from your dev box, via `tashpc-cursor`):

  ```bash
  python scripts/_launch_nemotron_sft_deepseek_remote.py              # default: forward profile
  python scripts/_launch_nemotron_sft_deepseek_remote.py --profile smoke
  python scripts/_launch_nemotron_sft_deepseek_remote.py --print-only  # show remote command
  ```

  The default **forward** profile uses ``MAX_LEN=512``, ``EPOCHS=1.0``, thinking template **on** (matches ``eval.greedy_harness``), and ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``. Smoke sets ``NEMOTRON_NO_THINKING=1`` for the shortest runs.

- **Shell entrypoint** on the server: `scripts/run_nemotron_sft_deepseek_synth_4gpu.sh`  
  Env: **`NEMOTRON_LORA=attn`** (default, `--lora-attn-fallback`) or **`mixer`** (`--peft-simple-targets`; optional **`MIXER_LORA_MAX_LAYERS`**), **`GRAD_ACCUM`**, **`MAX_LEN`**, **`EPOCHS`**, **`NEMOTRON_NO_THINKING=1`** to match old smoke, **`DATALOADER_NUM_WORKERS`**, omit **`MAX_SAMPLES`** for the full row count.

- **Why “accuracy” looked low:** `SFTTrainer` logs **token-level** metrics and **CE loss** on full teacher strings — they are not puzzle accuracy. Measure **`eval.greedy_harness`** accuracy instead.
- **Train/eval alignment:** Greedy eval uses **`enable_thinking=True`** in the chat template. Training now defaults to the same (omit `NEMOTRON_NO_THINKING`); pass `NEMOTRON_NO_THINKING=1` only for the tiniest smoke if needed.
- **Real holdout:** Rebuild with `python scripts/build_nemotron_sft_jsonl_from_deepseek_synth.py --holdout-fraction 0.12` → `train.jsonl` + `val_greedy.jsonl`, push both, retrain, then run vLLM eval on `val_greedy.jsonl`.
- **Quick sanity eval (same prompts as train):** `python scripts/build_eval_jsonl_from_nemotron_train.py --out data/cache/nemotron_eval/train_prompts_val.jsonl --every-n 3` then `python -m eval.greedy_harness --adapter … --val-jsonl …` on a GPU box (add `--dtype float16` on V100).
- **Holdout eval on the server:** after training, run Transformers eval (vLLM does not support Nemotron-H here) in a **persistent shell** or ``nohup`` so SSH does not drop mid-run::

  ```bash
  cd /home/jovyan/work/Nemotron-training
  nohup python3 scripts/eval_nemotron_holdout_transformers.py \
    --adapter outputs/nemotron_sft_deepseek_forward_v2 \
    --val-jsonl data/cache/nemotron_sft_deepseek/val_greedy.jsonl \
    --out-jsonl outputs/nemotron_sft_deepseek_forward_v2/eval_holdout.jsonl \
    --max-new-tokens 1024 \
    > outputs/nemotron_sft_deepseek_forward_v2/eval_transformers.log 2>&1 &
  tail -f outputs/nemotron_sft_deepseek_forward_v2/eval_transformers.log
  ```

  Then list failures::

  ```bash
  python3 scripts/summarize_eval_failures.py outputs/nemotron_sft_deepseek_forward_v2/eval_holdout.jsonl
  ```

- **Val mix (no GPU):** ``python scripts/analyze_val_prompt_mix.py data/cache/nemotron_sft_deepseek/val_greedy.jsonl`` — see which prompt families dominate the holdout.

- `teacher_high_rank.py` — Phase 3 rank-128 teacher.
- `svd_init.py` — SVD-truncate teacher ΔW to rank 32.
- `kd.py` — KD student training.

## Critical model facts

- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (MoE, 30B total /
  3B active). Locked by the competition.
- Architecture: **Mamba-Transformer hybrid** (Nemotron-H). Default LoRA targets
  (see `_lora.py`) follow **`docs/NEMOTRON_PLAN.md`**: mixer
  `in_proj` / `out_proj` on a fixed subset of layers — **not** the generic
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` set from
  vanilla Transformer examples.
- Public solution flagged a **known hang in Unsloth** on this model’s huge
  module count. This repo uses **PEFT + TRL directly** (no Unsloth wrapper).
