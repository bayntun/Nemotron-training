"""
Phase 1 - QLoRA SFT on Nemotron-3-Nano (competition base model).

Runs on **Linux + CUDA** (V100 server / JupyterHub). Not supported on CPU-only Windows dev box.

Example (single GPU, local SFT cache):

    python -m train.sft \\
      --output-dir ./outputs/sft_baseline

Example (4× GPU + DeepSpeed ZeRO-2):

    accelerate launch --config_file configs/accelerate_zero2.yaml \\
      train/sft.py --output-dir ./outputs/sft_baseline

Prerequisites: `pip install -r requirements.txt -r requirements-train.txt`, HF token,
accepted dataset ToS, and `python -m data.download --sft-only` unless using `--dataset-name`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument(
        "--model-name-or-path",
        type=str,
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        help="Base checkpoint (competition-locked default).",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data" / "cache" / "sft",
        help="Local HF snapshot from data.download (parquet/jsonl).",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional HF dataset id (overrides --dataset-dir).",
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Adapter + checkpoints.")
    p.add_argument("--max-samples", type=int, default=None, help="Debug cap on rows.")
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-strategy", type=str, default="epoch")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-init",
        type=str,
        default="pissa",
        help="PEFT init_lora_weights (default PiSSA per plan).",
    )
    p.add_argument(
        "--peft-simple-targets",
        action="store_true",
        help="Use target_modules in_proj/out_proj + layers_to_transform instead of regex.",
    )
    p.add_argument(
        "--target-modules",
        type=str,
        default=None,
        help="Comma-separated LoRA target module names (overrides default Nemotron targets).",
    )
    p.add_argument("--no-thinking-template", action="store_true", help="Disable enable_thinking in chat template.")
    p.add_argument("--attn-implementation", type=str, default="sdpa", choices=("sdpa", "eager"))
    p.add_argument("--optim", type=str, default="paged_adamw_8bit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load tokenizer + dataset only; print column stats and exit.",
    )
    return p.parse_args()


def _dry_run(args: argparse.Namespace) -> int:
    """Load tokenizer + dataset schema without CUDA or full model."""
    from transformers import AutoTokenizer

    from train._dataset import load_sft_raw, normalize_messages_dataset

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        token=hf_token,
    )
    ds = load_sft_raw(
        dataset_dir=None if args.dataset_name else args.dataset_dir,
        hub_id=args.dataset_name,
        hf_token=hf_token,
    )
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print("columns:", ds.column_names, "n=", len(ds), file=sys.stderr)
    ds = normalize_messages_dataset(ds)
    sample = ds[0]["messages"]
    print("dry-run OK — first messages roles:", [m.get("role") for m in sample])
    try:
        tpl_kw = {} if args.no_thinking_template else {"enable_thinking": True}
        text = tokenizer.apply_chat_template(
            sample,
            tokenize=False,
            add_generation_prompt=False,
            **tpl_kw,
        )
        print("chat_template chars:", len(text))
    except TypeError:
        text = tokenizer.apply_chat_template(
            sample,
            tokenize=False,
            add_generation_prompt=False,
        )
        print("chat_template chars:", len(text), "(enable_thinking unsupported)")
    return 0


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.dry_run:
        return _dry_run(args)

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from train._dataset import load_sft_raw, normalize_messages_dataset
    from train._lora import MAMBA_LAYER_INDICES, build_lora_config, mixer_lora_target_patterns

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required. Run on the Linux GPU server.", file=sys.stderr)
        return 1

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds = load_sft_raw(
        dataset_dir=None if args.dataset_name else args.dataset_dir,
        hub_id=args.dataset_name,
        hf_token=hf_token,
    )
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print("columns:", ds.column_names, "n=", len(ds), file=sys.stderr)

    ds = normalize_messages_dataset(ds)
    if not args.no_thinking_template:

        def _thinking_kw(example: dict) -> dict:
            out = dict(example)
            out["chat_template_kwargs"] = {"enable_thinking": True}
            return out

        ds = ds.map(_thinking_kw)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # In multi-process runs (accelerate/torchrun), pin each rank to a single GPU.
    # Using device_map="auto" under DDP can place both ranks on the same device.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device_map = {"": local_rank} if world_size > 1 else "auto"

    model_init_kw = dict(
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        token=hf_token,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_init_kw)

    custom_targets = None
    if args.target_modules:
        custom_targets = [x.strip() for x in args.target_modules.split(",") if x.strip()]

    if custom_targets:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=custom_targets,
            init_lora_weights=args.lora_init,
        )
    elif args.peft_simple_targets:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["in_proj", "out_proj"],
            layers_to_transform=list(MAMBA_LAYER_INDICES),
            init_lora_weights=args.lora_init,
        )
    else:
        peft_config = build_lora_config(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights=args.lora_init,
            target_modules=mixer_lora_target_patterns(),
        )

    # Nemotron-H: gradient checkpointing unsupported — TRL defaults True; force off.
    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        fp16=True,
        bf16=False,
        gradient_checkpointing=False,
        optim=args.optim,
        max_length=args.max_length,
        packing=False,
        seed=args.seed,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    print("Saved adapter under:", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
