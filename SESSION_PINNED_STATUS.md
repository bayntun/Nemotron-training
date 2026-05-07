# Session Pinned Status (2026-05-07)

This file captures the last known-good settings and next steps so work can resume quickly.

## Best current training settings

- Script: `tmp_train_csv_remote.py` (executed inside JupyterHub container)
- Core settings:
  - `max_length=512`
  - `per_device_batch_size=2`
  - `grad_accum=2`
  - `learning_rate=2e-4`

## Best result so far

- Run output: `/home/jovyan/work/Nemotron-training/outputs/csv_train_best_v1`
- Args:
  - `limit=3000`
  - `eval_size=300`
  - `epochs=1.5`
  - plus core settings above
- Holdout result: `accuracy=62/300=0.207`
- Artifacts:
  - Adapter: `/home/jovyan/work/Nemotron-training/outputs/csv_train_best_v1/adapter`
  - Metrics: `/home/jovyan/work/Nemotron-training/outputs/csv_train_best_v1/metrics.json`

## Tuning notes

- A (`max_length=384, bs=1, ga=4`) -> `0.160`
- B (`max_length=512, bs=1, ga=4`) -> `0.150`
- C (`max_length=512, bs=2, ga=2`) -> `0.160` and much faster (chosen)
- D (`max_length=1024, bs=4, ga=1`) -> `0.130` (more memory, worse quality)

## Throughput notes

- Synthetic 4xV100 benchmark reached much higher utilization than mini trainer.
- Real mini trainer showed bursty usage; future optimization path is true DDP (`torchrun`) and dataloader tuning.

## Planned next run

- `limit=6000`
- `eval_size=600`
- `epochs=2`
- Keep core settings: `max_length=512, bs=2, ga=2, lr=2e-4`

## Kaggle non-Mamba signal (RTX 6000, 2026-05-07 night)

- Kernel: `https://www.kaggle.com/code/bayntuna/rtx6000-nonmamba-train-v2`
- Script: `kaggle_nonmamba_train/run.py` (offline GPT-style baseline; no `mamba-ssm` dependency)
- Sweep summary (`nonmamba_summary.json`):
  - A (`max_length=384, bs=1, ga=4`) -> `ppl=1.389`, `~35k tok/s`
  - B (`max_length=512, bs=1, ga=4`) -> `ppl=1.284`, `~46k tok/s`
  - C (`max_length=512, bs=2, ga=2`) -> `ppl=1.283`, `~66k tok/s` (best overall)
- Transferable takeaway for V100 resume:
  - Keep `max_length=512`
  - Prefer higher per-device batch with lower grad-accum at same effective batch (C-shape config)
