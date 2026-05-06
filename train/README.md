# `train/`

Training scripts for SFT (Phase 1), high-rank teacher + KD (Phase 3), and
optionally GRPO (Phase 4, deprioritized).

## What will live here

- `sft.py` — Phase 1 baseline. LoRA `r=32` SFT on the cleaned andy279 data
  with PiSSA initialization. Runs on the 4x V100 server with QLoRA + DeepSpeed
  ZeRO-2 + gradient checkpointing + fp16 grad scaler (no bf16 on Volta).
- `teacher_high_rank.py` — Phase 3 step 1. Train a rank-128 LoRA teacher on
  the curated Phase 2 SFT mix.
- `svd_init.py` — Phase 3 step 2. Per-layer SVD-truncate the rank-128 teacher
  `\Delta W` to rank 32. Produces a student-init adapter directory.
- `kd.py` — Phase 3 step 3. KD-train the rank-32 student from the SVD init,
  using teacher's greedy traces (token-level KL on correct rollouts + small CE
  on the boxed-answer span).

## Critical model facts

- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (MoE, 30B total /
  3B active). Locked by the competition.
- Architecture: **Mamba-Transformer hybrid** (Nemotron-H). LoRA target_modules
  are `backbone.layers.{i}.mixer.in_proj` and
  `backbone.layers.{i}.mixer.out_proj` on the Mamba mixer layers, NOT the
  standard `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
  set you would use on a vanilla transformer.
- Public solution flagged a **known hang in Unsloth on this model's 5980
  modules**. Use `peft.get_peft_model(...)` directly with explicit
  `target_modules=MAMBA_MODULES`; do not wrap with `FastLanguageModel.get_peft_model`.

## Status

To be implemented in Phase 1 onward. This directory is intentionally empty
during Phase 0.
