# Scripts Guide (Active vs Archived)

Use this file to avoid ambiguity.

## Active (current operating mode)

- `scripts/_launch_nemotron_sft_deepseek_remote.py` — canonical remote train launch.
- `scripts/run_nemotron_sft_deepseek_synth_4gpu.sh` — canonical server-side train entrypoint.
- `scripts/_sync_nemotron_sft_to_remote.py` — canonical file sync to server container.
- `scripts/eval_nemotron_holdout_transformers.py` — canonical holdout eval (Nemotron-H compatible).
- `scripts/report_symbolic_holdout_eval.py` — symbolic subset success/failure report.

## Active utilities (keep, but non-canonical path)

- `scripts/mamba_ddp_smoke.py` — distributed smoke for stack sanity.
- `scripts/nemotron_shared_smoke.py` — quick shared-memory/path sanity checks.
- `scripts/nemotron_burn.py` — load/burn diagnostic utility for stability checks.
- `scripts/benchmark_gpu.py` — standalone GPU throughput benchmark helper.

## Archived (legacy experiments)

- `scripts/archive/2026-05-experiments/` — CSV/TinyLlama launcher history and older helpers.
- `archive/2026-05-experiments/` — root-level temp scripts moved out of top-level.

## Policy (applies to non-Python too)

- Archive both Python and non-Python legacy files (`.sh`, `.yaml`, etc.) when they are not canonical.
- Keep only active entrypoints in top-level `scripts/`.
- New throwaway scripts should not live in repo root; place them under `archive/` or a dedicated temp folder.
