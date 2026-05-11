#!/usr/bin/env bash
# Train TinyLlama CSV LoRA on all 4 V100s (DDP). Run inside the JupyterHub container, from repo root.
# Example (fresh LoRA):
#   docker exec jupyter-abayntun bash /home/jovyan/work/Nemotron-training/scripts/run_v100_csv_train_4gpu.sh \
#     --limit -1 --eval-size 600 --epochs 2 --max-length 512 \
#     --per-device-batch-size 2 --grad-accum 2 --learning-rate 2e-4 \
#     --output-dir /home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v2_full
#
# Continue from v2 adapter + push GPU harder (same effective batch size 16, fewer accum steps):
#   docker exec jupyter-abayntun bash /home/jovyan/work/Nemotron-training/scripts/run_v100_csv_train_4gpu.sh \
#     --adapter-in /home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v2_full/adapter \
#     --train-csv /home/jovyan/work/train.csv --limit -1 --eval-size 600 \
#     --epochs 1 --max-length 512 --per-device-batch-size 4 --grad-accum 1 \
#     --learning-rate 1e-4 --dataloader-num-workers 8 --save-steps 250 \
#     --output-dir /home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v3_continue_gpuheavy
#
# Optional: append least-squares hints (linear m→m′ or d=k·t²) to matching prompts:
#   ... tmp_train_csv_remote.py ... --inject-numeric-baseline auto
# Optional: length/word-count structural hints for `A -> B` cipher-style few-shot rows:
#   ... --inject-cipher-length-hint auto
set -euo pipefail
export TORCH_DISTRIBUTED_USE_LIBUV="${TORCH_DISTRIBUTED_USE_LIBUV:-0}"
REPO="${REPO:-/home/jovyan/work/Nemotron-training}"
cd "$REPO"
exec torchrun --standalone --nproc_per_node=4 tmp_train_csv_remote.py "$@"
