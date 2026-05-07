import csv
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def read_rows(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=2200)
    p.add_argument("--eval-size", type=int, default=200)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--output-dir", type=str, default="/home/jovyan/work/Nemotron-training/outputs/csv_tinyllama_mini")
    return p.parse_args()


def main() -> int:
    args_in = parse_args()
    train_csv = Path("/home/jovyan/work/train.csv")
    out_dir = Path(args_in.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_rows(train_csv, limit=args_in.limit)
    if len(train_rows) <= args_in.eval_size:
        raise RuntimeError("train.csv needs at least 12 rows for train/eval split")
    test_rows = train_rows[-args_in.eval_size :]
    train_rows = train_rows[: -args_in.eval_size]

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    train_text = [f"User: {r['prompt']}\nAssistant: {r['answer']}" for r in train_rows]
    ds = Dataset.from_dict({"text": train_text})

    def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        out = tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args_in.max_length,
        )
        out["labels"] = [x[:] for x in out["input_ids"]]
        return out

    tok_ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args_in.per_device_batch_size,
        gradient_accumulation_steps=args_in.grad_accum,
        num_train_epochs=args_in.epochs,
        learning_rate=args_in.learning_rate,
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )
    Trainer(model=model, args=args, train_dataset=tok_ds).train()
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))

    correct = 0
    for row in test_rows:
        prompt = row["prompt"]
        gt = str(row["answer"]).strip()
        inp = tok(f"User: {prompt}\nAssistant:", return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=24, do_sample=False, temperature=0.0)
        gen = tok.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        pred = gen.split()[0] if gen else ""
        ok = pred == gt
        correct += int(ok)
        print(json.dumps({"id": row.get("id"), "gt": gt, "pred": pred, "ok": ok}, ensure_ascii=False))

    total = len(test_rows)
    acc = correct / max(total, 1)
    metrics = {
        "train_rows": len(train_rows),
        "eval_rows": total,
        "max_length": args_in.max_length,
        "per_device_batch_size": args_in.per_device_batch_size,
        "grad_accum": args_in.grad_accum,
        "learning_rate": args_in.learning_rate,
        "accuracy": acc,
        "correct": correct,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"accuracy={correct}/{total}={acc:.3f}")
    print(f"metrics_saved={out_dir / 'metrics.json'}")
    print(f"adapter_saved={adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
