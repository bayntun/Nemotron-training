# Kaggle Execution Switch

This runbook switches training/eval work from the remote JupyterHub host to Kaggle notebooks.

## 1) Upload repo snapshot

From local repo root:

```powershell
New-Item -ItemType Directory -Force -Path dist | Out-Null
git archive --format=zip --prefix=Nemotron-training/ -o dist/nemotron-training-for-kaggle.zip HEAD
```

In Kaggle notebook UI:
- Add dataset/file containing `nemotron-training-for-kaggle.zip`
- Unzip into writable working dir:

```bash
mkdir -p /kaggle/working/src
unzip -q /kaggle/input/<your-upload>/nemotron-training-for-kaggle.zip -d /kaggle/working/src
cd /kaggle/working/src/Nemotron-training
```

## 2) Environment and auth

```bash
python -m pip install -U pip
pip install -r requirements.txt
pip install -r requirements-train.txt
```

Set secrets via Kaggle Secrets:
- `HF_TOKEN`
- `DEEPSEEK_API_KEY` (only if needed for teacher steps)

## 3) Data checks (competition CSVs)

```bash
ls -lh /kaggle/input/*/train.csv /kaggle/input/*/test.csv
```

## 4) Recommended first Kaggle run

Use the currently best settings from `SESSION_PINNED_STATUS.md`:

```bash
python /kaggle/working/src/Nemotron-training/tmp_train_csv_remote.py \
  --limit 3000 \
  --eval-size 300 \
  --epochs 1.5 \
  --max-length 512 \
  --per-device-batch-size 2 \
  --grad-accum 2 \
  --output-dir /kaggle/working/outputs/csv_train_best_v1
```

Then scale:

```bash
python /kaggle/working/src/Nemotron-training/tmp_train_csv_remote.py \
  --limit 6000 \
  --eval-size 600 \
  --epochs 2 \
  --max-length 512 \
  --per-device-batch-size 2 \
  --grad-accum 2 \
  --output-dir /kaggle/working/outputs/csv_train_best_v2
```

## 5) Notes

- `test.csv` in this workflow is typically unlabeled; evaluate on a holdout split from `train.csv`.
- Kaggle runtime limits apply; save artifacts under `/kaggle/working/outputs`.
- If GPU remains underutilized, increase `--max-length` and/or `--per-device-batch-size` carefully until near OOM.
