# JupyterHub on the training server

Use this when **training runs on your Linux GPU box** (e.g. 4× V100) and you access it through **JupyterHub** instead of SSH-only workflows.

The notebook **`bootstrap/remote_preflight.ipynb`** already supports JupyterHub paths (`/home/jovyan/...`), GPU checks, clone/verify, CONFIG, grader tests, and HF/DeepSeek smoke steps. Treat this doc as the **checklist** around that notebook.

## One-time setup

1. **Clone the repo** somewhere under your persistent home (typical Hub layouts use `/home/jovyan/work`):

   ```bash
   cd ~/work   # or your hub’s project directory
   git clone https://github.com/bayntun/Nemotron-training.git
   cd Nemotron-training
   ```

2. **Python environment** (venv in-repo keeps paths predictable):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip wheel
   pip install -r requirements.txt
   pip install -r requirements-train.txt
   ```

   Volta (V100): stay on **fp16** + grad scaler for training; see comments in [`requirements-train.txt`](../requirements-train.txt).

3. **Register the kernel** (optional but convenient):

   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name=nemotron --display-name="Python (nemotron)"
   ```

   In JupyterHub, open `bootstrap/remote_preflight.ipynb` and select **Python (nemotron)**.

## Secrets

Do **not** commit real tokens. Prefer hub-provided env vars **`HF_TOKEN`** and **`DEEPSEEK_API_KEY`** if your admin injects them into single-user servers.

Otherwise, on the server:

```bash
cp bootstrap/secrets_local.env.example bootstrap/secrets_local.env
# edit bootstrap/secrets_local.env
```

The CONFIG paths in `remote_preflight.ipynb` pick up `bootstrap/secrets_local.env` relative to `NEMOTRON_REPO` when set.

## Validation flow (first session)

1. Start JupyterHub → open **`bootstrap/remote_preflight.ipynb`** from the clone.
2. Run **system sanity** (GPU / torch).
3. **Skip** Kaggle-only cells (`/kaggle/input`, dataset ZIP) if you are **not** on Kaggle.
4. **Git clone** cell: skip if the repo is already this checkout; or point clone URL at your fork.
5. Run **CONFIG** / env checks, then **`pytest`** / grader cells as listed in the notebook.
6. When Phase 0 data steps are ready: in a terminal with venv activated, run `python -m data.download` and `python -m teacher.smoke_test`.

## Training vs notebooks

- **Phase 1** training (`train/sft.py`, etc.) will normally run from a **terminal** on the same machine (`torchrun`, DeepSpeed). JupyterHub is for exploration, preflight, and light debugging—not the only way to launch multi-GPU jobs.
- If your hub **does not** expose a terminal or long-running processes time out, use **SSH** or **batch jobs** for multi-hour training and keep Jupyter for notebooks only.

## Hugging Face cache

Large models fill disk quickly. If the hub provides shared fast storage, set once (example):

```bash
export HF_HOME=/path/to/scratch/huggingface
```

Put that in your shell profile on the server or hub **environment** so notebooks and CLI agree.

## See also

- [`bootstrap/README.md`](../bootstrap/README.md) — secrets file layout  
- [`docs/NEMOTRON_PLAN.md`](NEMOTRON_PLAN.md) — Phase 1 milestones  
- [`docs/GPU_SYNTH_BENCHMARK.md`](GPU_SYNTH_BENCHMARK.md) — optional throughput sanity on this hardware  
