# Kaggle Local Mamba Wheels

Use this when Kaggle CLI workers cannot resolve PyPI/HF DNS.

## Required wheel files

Attach a Kaggle dataset containing at least:

- `causal_conv1d-*.whl`
- `mamba_ssm-*.whl` (or `mamba-ssm-*.whl`)

The training kernel script `kaggle_rtx_train/run.py` now installs only from local `.whl` files under `/kaggle/input`.

## Recommended way to build wheelhouse

Build wheels in a Linux CUDA environment (same Python minor version as Kaggle runtime), then upload wheel files as a private Kaggle dataset.

Example high-level flow:

1. In a Linux CUDA environment, install compatible torch first.
2. Build/download `causal-conv1d` wheel.
3. Build/download `mamba-ssm` wheel.
4. Upload all `.whl` files as a Kaggle dataset named e.g. `bayntuna/mamba-wheelhouse`.
5. Attach that dataset to the kernel.

## What the kernel expects

At runtime, script searches `/kaggle/input/**` for `.whl` files and filters by names containing:

- `causal_conv1d`
- `mamba_ssm` or `mamba-ssm`

If missing, it fails fast with a clear message.
