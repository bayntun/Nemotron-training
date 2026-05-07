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
accelerate launch --config_file configs/accelerate_zero2.yaml train/sft.py \
  --output-dir ./outputs/sft_baseline
```

**Packaging:** after training, point `python -m submit.package --adapter <output-dir>` at the saved adapter directory (once weights exist).

### Flags worth knowing

| Flag | Purpose |
|------|--------|
| `--peft-simple-targets` | Use `in_proj`/`out_proj` + `layers_to_transform` instead of regex targets (if regex matches nothing). |
| `--no-thinking-template` | Omit `enable_thinking` in chat-template kwargs. |
| `--max-samples N` | Smoke subset. |
| `--attn-implementation eager` | If `sdpa` misbehaves on your stack. |

**Nemotron-H:** keep **`gradient_checkpointing=False`** (unsupported / hangs). TRL’s `SFTConfig` turns this off explicitly.

### Still to implement

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
