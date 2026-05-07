# Bootstrap

- **`remote_preflight.ipynb`** — run on **JupyterHub** (training server) or **Kaggle** to validate GPU, repo layout, grader tests, and HF + DeepSeek smoke checks.
- **Server checklist:** [docs/JUPYTERHUB.md](../docs/JUPYTERHUB.md) (venv, kernel, secrets, which cells apply off-Kaggle).

## Secrets

Do not paste API keys into notebook cells you commit. GitHub **secret scanning** will block the push.

1. **Preferred:** set `HF_TOKEN` and `DEEPSEEK_API_KEY` in the Jupyter/container environment.
2. **Or:** copy `secrets_local.env.example` to **`secrets_local.env`** (gitignored) under this folder and fill in values on each machine.

The CONFIG cell loads `bootstrap/secrets_local.env` from your `NEMOTRON_REPO` if present.
