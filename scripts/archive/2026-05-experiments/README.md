# Archive: 2026-05 Experiments

This folder stores legacy experiment launchers and helper scripts that are **not**
canonical for the current operating mode.

## Why archived

- Preserves reproducibility of CSV/TinyLlama-era experiments.
- Keeps top-level `scripts/` focused on active Nemotron contest workflows.
- Includes both Python and non-Python files (for example `.sh`) when they are
  tied to legacy flows.

## Replaced by (current canonical flow)

- Remote Nemotron train launch: `scripts/_launch_nemotron_sft_deepseek_remote.py`
- Remote sync bundle: `scripts/_sync_nemotron_sft_to_remote.py`
- Holdout eval (Transformers): `scripts/eval_nemotron_holdout_transformers.py`
- Symbolic report: `scripts/report_symbolic_holdout_eval.py`

## Contents

- CSV experiment launchers: `_launch_csv_v3...v13_*.py`
- CSV sync helper: `_sync_csv_trainer_to_remote.py`
- CSV 4-GPU shell entrypoint: `run_v100_csv_train_4gpu.sh`
- one-off data probe: `_dump_eval_cipher_sample.py`
- legacy CSV analysis/build helpers:
  - `analyze_eval_failures.py`
  - `build_sft_from_csv.py`
  - `inspect_train_csv_numeric.py`
  - `merge_synth_train_csv.py`
  - `push_merged_train_csv_remote.py`

If a historical run must be reproduced, copy the needed script back to `scripts/`
(or invoke it directly from this archive path).
