import csv
import json
import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def read_rows(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit < 0:
        return rows
    return rows[:limit]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=str, default="/home/jovyan/work/train.csv")
    p.add_argument(
        "--limit",
        type=int,
        default=2200,
        help="Max CSV rows (-1 = use full file; avoids edge cases with 0 in some shells).",
    )
    p.add_argument("--eval-size", type=int, default=200)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lr-scheduler-type", type=str, default="cosine")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=None, help="default: 2 * lora_r")
    p.add_argument(
        "--adapter-in",
        type=str,
        default="",
        help="Path to existing PEFT adapter dir to continue training (empty = new LoRA).",
    )
    p.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help=(
            "HF Trainer checkpoint directory (e.g. .../checkpoint-500). "
            "Use after an interrupted run that saved checkpoints under --output-dir; "
            "do not combine with --adapter-in."
        ),
    )
    p.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="If >0, save checkpoints every N steps (enables resume). Default 0 = no mid-run saves.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
        help="Greedy generation cap for eval (raise for multi-word / long cipher answers).",
    )
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="/home/jovyan/work/Nemotron-training/outputs/csv_tinyllama_mini")
    p.add_argument(
        "--inject-numeric-baseline",
        type=str,
        default="none",
        choices=("none", "auto", "linear_meters", "gravity_quadratic"),
        help=(
            "Append a short hint derived from the listed example pairs: "
            "linear least-squares for 'unit conversion' m→m′ rows, or d=k·t² for gravity tables. "
            "Applied to User text in both training and eval; not ground truth if the hidden rule differs."
        ),
    )
    p.add_argument(
        "--inject-cipher-length-hint",
        type=str,
        default="none",
        choices=("none", "auto"),
        help=(
            "Append a structural hint when the prompt has multiple length- and word-count-preserving "
            "`A -> B` examples and a parsable query string (cipher / string-transform family)."
        ),
    )
    return p.parse_args()


def use_ddp() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def main() -> int:
    args_in = parse_args()
    from train.csv_cipher_length_hint import augment_prompt_for_cipher_length_hint
    from train.csv_numeric_baseline import augment_prompt_for_numeric_baseline

    strat = args_in.inject_numeric_baseline
    cipher_strat = args_in.inject_cipher_length_hint
    train_csv = Path(args_in.train_csv)
    out_dir = Path(args_in.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_rows(train_csv, limit=args_in.limit)
    if len(train_rows) <= args_in.eval_size:
        raise RuntimeError("train.csv needs at least 12 rows for train/eval split")
    test_rows = train_rows[-args_in.eval_size :]
    train_rows = train_rows[: -args_in.eval_size]
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(
            json.dumps(
                {
                    "train_csv": str(train_csv),
                    "limit_arg": args_in.limit,
                    "train_rows": len(train_rows),
                    "eval_rows": len(test_rows),
                    "inject_numeric_baseline": strat,
                    "inject_cipher_length_hint": cipher_strat,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ddp = use_ddp()
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        base = base.to(local_rank)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    resume_ckpt = (args_in.resume_from_checkpoint or "").strip()
    if resume_ckpt and args_in.adapter_in.strip():
        raise RuntimeError("Use either --resume-from-checkpoint OR --adapter-in, not both.")

    lora_alpha = args_in.lora_alpha if args_in.lora_alpha is not None else 2 * args_in.lora_r
    if args_in.adapter_in:
        adapter_path = Path(args_in.adapter_in)
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"--adapter-in not found: {adapter_path}")
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=args_in.lora_r,
                lora_alpha=lora_alpha,
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

    def _pack_row(r: dict[str, str]) -> str:
        p = augment_prompt_for_numeric_baseline(r["prompt"], strat)  # type: ignore[arg-type]
        p = augment_prompt_for_cipher_length_hint(p, cipher_strat)  # type: ignore[arg-type]
        return f"User: {p}\nAssistant: {r['answer']}"

    train_text = [_pack_row(r) for r in train_rows]
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

    use_cuda = torch.cuda.is_available()
    train_kw: dict = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=args_in.per_device_batch_size,
        gradient_accumulation_steps=args_in.grad_accum,
        num_train_epochs=args_in.epochs,
        learning_rate=args_in.learning_rate,
        lr_scheduler_type=args_in.lr_scheduler_type,
        warmup_ratio=args_in.warmup_ratio,
        weight_decay=args_in.weight_decay,
        max_grad_norm=args_in.max_grad_norm,
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=args_in.dataloader_num_workers,
        dataloader_pin_memory=use_cuda,
        dataloader_drop_last=ddp,
        ddp_find_unused_parameters=False,
    )
    if args_in.dataloader_num_workers > 0:
        train_kw["dataloader_prefetch_factor"] = 2
    if args_in.save_steps > 0:
        train_kw["save_strategy"] = "steps"
        train_kw["save_steps"] = int(args_in.save_steps)
        train_kw["save_total_limit"] = 3
    args = TrainingArguments(**train_kw)
    trainer = Trainer(model=model, args=args, train_dataset=tok_ds)
    trainer.train(resume_from_checkpoint=(resume_ckpt if resume_ckpt else None))

    if dist.is_initialized():
        dist.barrier()

    if trainer.is_world_process_zero():
        eval_model = unwrap_model(trainer.model)
        eval_model.eval()
        dev = next(eval_model.parameters()).device

        adapter_dir = out_dir / "adapter"
        eval_model.save_pretrained(str(adapter_dir))

        def norm_first_line(s: str) -> str:
            return s.split("\n", 1)[0].strip()

        eval_lines: list[str] = []
        correct_full = 0
        correct_first = 0
        for row in test_rows:
            prompt = augment_prompt_for_numeric_baseline(row["prompt"], strat)  # type: ignore[arg-type]
            prompt = augment_prompt_for_cipher_length_hint(prompt, cipher_strat)  # type: ignore[arg-type]
            gt = str(row["answer"]).strip()
            gt_first = (gt.split()[0] if gt else "").strip()
            inp = tok(f"User: {prompt}\nAssistant:", return_tensors="pt").to(dev)
            out = eval_model.generate(**inp, max_new_tokens=args_in.max_new_tokens, do_sample=False)
            gen = tok.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            gen_line = norm_first_line(gen)
            # `pred` is first whitespace token only (for ok_first); full line is `gen_line` / ok_full.
            pred = gen_line.split()[0] if gen_line else ""
            ok_full = gen_line == gt
            ok_first = bool(gt_first) and pred == gt_first
            correct_full += int(ok_full)
            correct_first += int(ok_first)
            rec = {
                "id": row.get("id"),
                "prompt": row.get("prompt", "")[:2000],
                "gt": gt,
                "pred": pred,
                "gen_full": gen_line,
                "ok_full": ok_full,
                "ok_first": ok_first,
            }
            eval_lines.append(json.dumps(rec, ensure_ascii=False))
            print(json.dumps({"id": row.get("id"), "gt": gt, "pred": pred, "ok_full": ok_full, "ok_first": ok_first}, ensure_ascii=False))

        total = len(test_rows)
        acc_full = correct_full / max(total, 1)
        acc_first = correct_first / max(total, 1)
        (out_dir / "eval_details.jsonl").write_text("\n".join(eval_lines) + "\n", encoding="utf-8")
        metrics = {
            "train_rows": len(train_rows),
            "eval_rows": total,
            "max_length": args_in.max_length,
            "per_device_batch_size": args_in.per_device_batch_size,
            "grad_accum": args_in.grad_accum,
            "learning_rate": args_in.learning_rate,
            "ddp_world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "dataloader_num_workers": args_in.dataloader_num_workers,
            "adapter_in": args_in.adapter_in or None,
            "lora_r": args_in.lora_r,
            "inject_numeric_baseline": strat,
            "inject_cipher_length_hint": cipher_strat,
            "accuracy_full": acc_full,
            "correct_full": correct_full,
            "accuracy_first": acc_first,
            "correct_first": correct_first,
            "accuracy": acc_full,
            "correct": correct_full,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"accuracy_full={correct_full}/{total}={acc_full:.3f} accuracy_first={correct_first}/{total}={acc_first:.3f}")
        print(f"metrics_saved={out_dir / 'metrics.json'}")
        print(f"adapter_saved={adapter_dir}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
