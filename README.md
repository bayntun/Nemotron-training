# Nemotron Reasoning Challenge — DGX Spark Plan

Submission pipeline for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).

**Goal:** win at least one DGX Spark via either Path B (top 10% +
Best Data/Synthetic Data Method Open Contribution Award) or Path A (top 5-8
final leaderboard via cascade for solo competitors).

**Current operating mode:** [docs/OPERATING_MODE.md](docs/OPERATING_MODE.md)  
(this tells future AI whether we are still experimental or in submission mode).

**Plan document (in-repo):** [docs/NEMOTRON_PLAN.md](docs/NEMOTRON_PLAN.md).

## Repo layout

```text
.
|-- data/        # andy279 dataset download scripts (cached files gitignored)
|-- solvers/     # Phase 2 per-category rule-discovery solvers (Phase 2)
|-- train/       # SFT (Phase 1), high-rank teacher + KD (Phase 3)
|-- eval/        # Verbatim Kaggle grader port + vLLM greedy harness
|-- teacher/     # DeepSeek V3.2 client for synthetic CoT generation
|-- submit/      # submission.zip packaging with pre-zip validation
|-- notebook/    # Public Kaggle notebook source (Path B deliverable)
|-- configs/     # Training/eval configs (per-phase)
|-- scripts/     # Run scripts and Phase orchestration
```

## Hard constraints (binding for evaluation)

- Base: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (Mamba-Transformer
  hybrid MoE, 30B total / 3B active).
- vLLM serving: `max_lora_rank=32`, `max_model_len=8192`, `max_tokens=7680`,
  `max_num_seqs=64`, `gpu_memory_utilization=0.85`.
- **`temperature=0.0, top_p=1.0`** — strictly greedy decode. No
  self-consistency, no test-time sampling.
- Submission: `submission.zip` containing `adapter_config.json` and
  `adapter_model.safetensors` at the zip root.
- Grader: `extract_final_answer` (boxed-first, heuristic fallbacks)
  followed by `math.isclose(rel_tol=1e-2, abs_tol=1e-5)` numeric path or
  case-insensitive string fallback.
- Public Kaggle notebook + write-up are mandatory for prize eligibility.

## Compute

- **Local Windows dev box** (this machine): orchestration, data prep,
  solver code, eval harness writing, packaging. No GPU work.
- **4x Tesla V100 Linux server** (16 GB or 32 GB per GPU depending on box):
  Phase 1 SFT, Phase 2 retrains, Phase 3 high-rank teacher + KD.
  **fp16 only** on Volta (no bf16). Primary access via **JupyterHub**:
  see [docs/JUPYTERHUB.md](docs/JUPYTERHUB.md) and
  `bootstrap/remote_preflight.ipynb`.
- **Kaggle Blackwell 96 GB notebook (30 hr/week)**: kernel-identical eval,
  `submission.zip` building, public notebook hosting. Reserved for
  Kaggle-only work; fallback if MoE-on-V100 blocks training (see plan).

## Setup (Windows dev box)

1. Install Python 3.12 (already present).
2. Create a venv:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in:

   ```text
   HF_TOKEN=hf_...
   DEEPSEEK_API_KEY=sk-...
   ```

4. Accept the click-through ToS on each gated HuggingFace dataset:
   - <https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge>
   - <https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces>

## Quickstart (Phase 0 verification)

```powershell
# 1. Verify the local grader matches the Kaggle kernel.
python -m pytest eval/test_grader.py -v

# 2. Download the andy279 datasets (one-time, ~1.4 GB total).
python -m data.download

# 3. Smoke-test the DeepSeek API (~$0.001).
python -m teacher.smoke_test

# 4. Sanity-check submission packaging on a dummy adapter dir
#    (after Phase 1 produces one).
python -m submit.package --adapter ./adapters/sft_baseline --out submission.zip
```

## Phase status

- [x] Phase 0: foundation, grader port, scaffolding (**repo commit `e81f24f`**; HF download + Kaggle smoke still open — see [docs/NEMOTRON_PLAN.md](docs/NEMOTRON_PLAN.md))
- [ ] Phase 1: SFT baseline + leaderboard milestone (**`train/sft.py`** + DeepSpeed configs in repo — run and tune on GPU server) — see [train/README.md](train/README.md)
- [ ] Phase 2: solver-guided synthetic data (centerpiece, both paths)
- [ ] Phase 3: rank-128 teacher -> SVD-init -> KD into rank-32 student (Path A booster)
- [ ] Phase 4: public notebook + write-up + final submission (Path B deliverable)
