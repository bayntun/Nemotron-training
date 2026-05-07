import csv
import json
import os
import random
import argparse
from pathlib import Path
import subprocess

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def find_train_csv() -> Path:
    paths = sorted(Path("/kaggle/input").glob("**/train.csv"))
    if not paths:
        raise FileNotFoundError("train.csv not found under /kaggle/input")
    return paths[0]


def find_local_model_dir() -> Path:
    # Prefer Kaggle-attached model assets to avoid external HF DNS/network dependency.
    candidates = sorted(Path("/kaggle/input").glob("**/config.json"))
    if not candidates:
        raise FileNotFoundError("No local model config.json found under /kaggle/input")
    return candidates[0].parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")
    return p.parse_args()


def main() -> None:
    args_cli = parse_args()
    random.seed(42)
    torch.manual_seed(42)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    print("installing_mamba_ssm_from_local_wheels...", flush=True)
    wheel_dirs = sorted(Path("/kaggle/input").glob("**/*mamba*wheel*"))
    wheel_files = []
    for d in wheel_dirs:
        wheel_files.extend(sorted(d.glob("*.whl")))
    # Fallback: any wheel drop under kaggle input (explicitly filtered later)
    if not wheel_files:
        for d in Path("/kaggle/input").glob("**/*"):
            if d.is_dir():
                wheel_files.extend(sorted(d.glob("*.whl")))

    causal = [str(p) for p in wheel_files if "causal_conv1d" in p.name]
    mamba = [str(p) for p in wheel_files if "mamba_ssm" in p.name or "mamba-ssm" in p.name]
    if not causal or not mamba:
        raise RuntimeError(
            "Local wheel install requested but missing required wheels. "
            "Attach a Kaggle dataset containing both causal-conv1d and mamba-ssm .whl files."
        )

    install_cmd = ["python", "-m", "pip", "install"] + causal + mamba + ["--no-deps"]
    install = subprocess.run(install_cmd, capture_output=True, text=True)
    print(f"mamba_local_install_returncode={install.returncode}", flush=True)
    if install.stdout:
        print("mamba_local_install_stdout_tail=", flush=True)
        print("\n".join(install.stdout.splitlines()[-20:]), flush=True)
    if install.stderr:
        print("mamba_local_install_stderr_tail=", flush=True)
        print("\n".join(install.stderr.splitlines()[-40:]), flush=True)
    if install.returncode != 0:
        raise RuntimeError("Failed local wheel install for mamba-ssm/causal-conv1d")

    subprocess_ok = os.system("nvidia-smi") == 0
    print(f"nvidia_smi_ok={subprocess_ok}", flush=True)
    if torch.cuda.is_available():
        print(
            f"gpu={torch.cuda.get_device_name(0)} cuda={torch.version.cuda} "
            f"torch={torch.__version__}",
            flush=True,
        )

    train_csv = find_train_csv()
    rows = list(csv.DictReader(train_csv.open("r", encoding="utf-8")))
    rows = [r for r in rows if "prompt" in r and "answer" in r]
    rows = rows[:3000]
    eval_size = 300
    train_rows = rows[:-eval_size]
    eval_rows = rows[-eval_size:]
    print(
        f"train_csv={train_csv} total_rows={len(rows)} train_rows={len(train_rows)} eval_rows={len(eval_rows)}",
        flush=True,
    )

    model_dir = find_local_model_dir()
    model_id = str(model_dir)
    print(f"model_dir={model_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    ds = Dataset.from_dict({"text": [f"User: {r['prompt']}\nAssistant: {r['answer']}" for r in train_rows]})

    def tok_fn(batch):
        out = tok(batch["text"], truncation=True, padding="max_length", max_length=512)
        out["labels"] = [x[:] for x in out["input_ids"]]
        return out

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
    args = TrainingArguments(
        output_dir="/kaggle/working/rtx_train_out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1.5,
        learning_rate=2e-4,
        fp16=True,
        logging_strategy="steps",
        logging_steps=20 if args_cli.quiet else 5,
        save_strategy="no" if args_cli.quiet else "steps",
        save_steps=500 if args_cli.quiet else 100,
        report_to=[],
    )
    print(
        "training_config="
        + json.dumps(
            {
                "batch": 2,
                "grad_accum": 2,
                "epochs": 1.5,
                "max_length": 512,
                "verbose": not args_cli.quiet,
            }
        ),
        flush=True,
    )
    Trainer(model=model, args=args, train_dataset=ds_tok).train()

    correct = 0
    for i, r in enumerate(eval_rows, start=1):
        inp = tok(f"User: {r['prompt']}\nAssistant:", return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=24, do_sample=False, temperature=0.0)
        pred = tok.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True).strip().split()
        pred0 = pred[0] if pred else ""
        correct += int(pred0 == str(r["answer"]).strip())
        if not args_cli.quiet and i % 25 == 0:
            print(f"eval_progress={i}/{len(eval_rows)} running_acc={correct/i:.4f}", flush=True)
    acc = correct / len(eval_rows)

    summary = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "accuracy": acc,
        "correct": correct,
        "model": model_id,
    }
    out = Path("/kaggle/working/rtx_train_out")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    model.save_pretrained(str(out / "adapter"))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
