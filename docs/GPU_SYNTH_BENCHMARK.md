# Synthetic training GPU benchmark (`scripts/benchmark_gpu.py`)

Micro-benchmark: Transformer-style blocks, forward + backward + AdamW step. **Not** Nemotron throughput; useful only for **relative** comparison when the **same CLI flags** are used on two machines.

**Notebook:** `notebook/v100_synthetic_benchmark.ipynb` — same hyperparameters, self-contained (no secrets or env files). Use **fp16** when comparing to the tables below.

## V100 server (measured)

**Hardware (from preflight):** 4× Tesla V100-SXM2-**16GB**, CUDA 12.2 / driver 535.x (exact PyTorch build not recorded).

**Command-line shape (held constant for cross-machine comparison):**

```bash
python scripts/benchmark_gpu.py \
  --batch 4 --seq 256 --hidden 3072 --layers 10 \
  --warmup 5 --steps 25 --dtype fp16
```

**4× V100 (DDP):**

| Metric | Value |
|--------|-------|
| `avg_step_s` | 0.110837 |
| `tokens_per_sec` | 36955.16 |
| `approx_tflops` | 55.80 |

**1× V100 (single process):**

| Metric | Value |
|--------|-------|
| `avg_step_s` | 0.090913 |
| `tokens_per_sec` | 11263.47 |
| `approx_tflops` | 17.01 |

**Scaling (4× vs 1× on this workload):** ~**3.28×** global tokens/sec (communication overhead shows up in DDP step time).

**DDP invocation:**

```bash
torchrun --nproc_per_node=4 scripts/benchmark_gpu.py \
  --batch 4 --seq 256 --hidden 3072 --layers 10 \
  --warmup 5 --steps 25 --dtype fp16
```

## Kaggle RTX PRO 6000 Blackwell (measured)

**Hardware:** NVIDIA RTX PRO 6000 **Blackwell Server Edition**, Kaggle competition GPU notebook, **`world_size=1`** (typical Jupyter single process).

**Notebook:** `notebook/v100_synthetic_benchmark.ipynb`, `DTYPE="fp16"`, same batch/seq/hidden/layers/warmup/steps as above.

| Metric | Value |
|--------|-------|
| `avg_step_s` | 0.039579 |
| `tokens_per_sec` | 25872.24 |
| `approx_tflops` | 39.07 |

Optional: retry `DTYPE = "bf16"` after the fp16 baseline if `torch.cuda.is_bf16_supported()` is true on that kernel.

### Cross-machine notes (same micro-benchmark only)

- **1× Blackwell vs 1× V100:** `tokens_per_sec` ≈ **25872 / 11263 ≈ 2.30×** (Blackwell higher on this toy workload).
- **1× Blackwell vs 4× V100 (DDP):** global `tokens_per_sec` ≈ **25872 / 36955 ≈ 0.70×** — the **four-GPU** V100 box still wins on this metric because `tokens_per_sec` scales by `WORLD_SIZE` in the script.
- **Training real models:** this benchmark does **not** rank “which machine is better for Nemotron training.” Use memory fit (e.g. 16 GB × 4 vs one large GPU), actual step time with your training stack, and where you need to run (Kaggle vs dedicated server).
