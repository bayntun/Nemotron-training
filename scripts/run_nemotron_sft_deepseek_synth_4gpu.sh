#!/usr/bin/env bash
# Nemotron-3-Nano QLoRA via train/sft.py + DeepSeek teacher JSONL dataset.
# Run inside the Jupyter container from repo root.
# LoRA targets: default ``NEMOTRON_LORA=attn`` (``--lora-attn-fallback``). Set
# ``NEMOTRON_LORA=mixer`` for ``--peft-simple-targets`` (+ optional ``MIXER_LORA_MAX_LAYERS``).
#
# Prerequisites on the server:
#   pip install -r requirements.txt -r requirements-train.txt
#   HF_TOKEN in env (or bootstrap/secrets_local.env synced into the container).
#   Dataset dir contains train.jsonl (see scripts/build_nemotron_sft_jsonl_from_deepseek_synth.py).
#
# Example (short smoke):
#   MAX_SAMPLES=16 EPOCHS=0.1 MAX_LEN=256 docker exec jupyter-abayntun bash \
#     /home/jovyan/work/Nemotron-training/scripts/run_nemotron_sft_deepseek_synth_4gpu.sh
#
# Forward run (full jsonl, no MAX_SAMPLES — omit the var):
#   EPOCHS=0.25 MAX_LEN=1024 GRAD_ACCUM=16 OUTPUT_DIR=.../nemotron_sft_forward docker exec ... bash scripts/...
#
# Competition-style mixer LoRA (try after 4bit smoke works; may need stack pins):
#   NEMOTRON_LORA=mixer MIXER_LORA_MAX_LAYERS=8 EPOCHS=0.2 MAX_LEN=512 docker exec ... bash scripts/...
#
# Env knobs:
#   NEMOTRON_LORA=attn (default) | mixer
#   GRAD_ACCUM — gradient accumulation steps (default 8)
#   ACCELERATE_CONFIG — default single-GPU fp16 yaml
#   NEMOTRON_NO_THINKING=1 — pass ``--no-thinking-template`` (matches tiny smoke; **off** by default
#     so training aligns with ``eval.greedy_harness``, which uses ``enable_thinking=True``).
#   DATALOADER_NUM_WORKERS — passed through (default 0 in train.sft)
set -eu
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
REPO="${REPO:-/home/jovyan/work/Nemotron-training}"
cd "$REPO"

DATASET_DIR="${DATASET_DIR:-/home/jovyan/work/Nemotron-training/data/cache/nemotron_sft_deepseek}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/work/Nemotron-training/outputs/nemotron_sft_deepseek_synth}"
MAX_LEN="${MAX_LEN:-4096}"
EPOCHS="${EPOCHS:-0.3}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NEMOTRON_LORA="${NEMOTRON_LORA:-attn}"

EXTRA=()
if [[ -n "${MAX_SAMPLES}" ]]; then EXTRA+=(--max-samples "${MAX_SAMPLES}"); fi

LORA_FLAGS=(--lora-attn-fallback)
if [[ "${NEMOTRON_LORA}" == "mixer" ]]; then
  LORA_FLAGS=(--peft-simple-targets)
  if [[ -n "${MIXER_LORA_MAX_LAYERS:-}" ]]; then
    LORA_FLAGS+=(--mixer-lora-max-layers "${MIXER_LORA_MAX_LAYERS}")
  fi
fi

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_single_gpu_fp16.yaml}"

THINK_ARGS=()
if [[ "${NEMOTRON_NO_THINKING:-}" == "1" ]]; then
  THINK_ARGS=(--no-thinking-template)
fi

DLNW="${DATALOADER_NUM_WORKERS:-0}"
DL_ARGS=()
if [[ "${DLNW}" != "0" ]]; then
  DL_ARGS=(--dataloader-num-workers "${DLNW}")
fi

exec accelerate launch --config_file "${ACCELERATE_CONFIG}" -m train.sft \
  --attn-implementation eager \
  "${LORA_FLAGS[@]}" \
  "${THINK_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-dir "${DATASET_DIR}" \
  --num-train-epochs "${EPOCHS}" \
  --max-length "${MAX_LEN}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --learning-rate 2e-4 \
  --lora-r 32 \
  --lora-init gaussian \
  "${DL_ARGS[@]}" \
  "${EXTRA[@]}"
