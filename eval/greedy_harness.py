"""
Local greedy-eval harness that mirrors the Kaggle scoring kernel exactly.

Runs on the V100 server (or Kaggle Blackwell) -- not Windows. Requires vLLM
and the Nemotron base model weights to be downloadable.

Kernel parameters (locked to the competition's evaluation kernel):
    max_lora_rank       = 32
    max_tokens          = 7680
    top_p               = 1.0
    temperature         = 0.0     (greedy decode -- no sampling)
    max_num_seqs        = 64
    gpu_memory_utilization = 0.85
    max_model_len       = 8192

USAGE:
    python -m eval.greedy_harness \\
        --base-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
        --adapter ./adapters/sft_baseline \\
        --val-jsonl ./data/cache/sft/validation.jsonl \\
        --out ./outputs/eval_sft_baseline.jsonl

Outputs a JSONL where each line is:
    {"id": ..., "prompt": ..., "ground_truth": ..., "generation": ...,
     "extracted": ..., "is_correct": bool}

Then prints overall accuracy and a per-category breakdown.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from eval.grader import extract_final_answer, verify

# Locked to the competition's vLLM serving config.
KERNEL_PARAMS = {
    "max_lora_rank": 32,
    "max_model_len": 8192,
    "max_num_seqs": 64,
    "gpu_memory_utilization": 0.85,
}
KERNEL_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 7680,
}


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _format_prompt(tokenizer, user_content: str, enable_thinking: bool = True) -> str:
    """Apply the chat template the same way the public solution does."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )


def _iter_examples(records: Iterable[dict]) -> Iterable[tuple[str, str, str, str]]:
    """Yield (id, prompt, ground_truth, category) per record."""
    for r in records:
        rid = str(r.get("id", ""))
        prompt = r.get("prompt") or r.get("question") or ""
        gt = str(r.get("ground_truth") or r.get("answer") or "")
        category = str(r.get("category", ""))
        if prompt and gt:
            yield rid, prompt, gt, category


def run_eval(
    base_model: str,
    adapter_dir: Path | None,
    records: list[dict],
    out_path: Path,
) -> dict:
    """Greedy decode every record and score with the local grader. Returns summary."""
    # Heavy imports kept inside the function so this module is importable on
    # boxes without vLLM (Windows dev box).
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"Loading tokenizer for {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"Initializing vLLM with kernel-identical params: {KERNEL_PARAMS}")
    llm = LLM(
        model=base_model,
        enable_lora=adapter_dir is not None,
        trust_remote_code=True,
        dtype="bfloat16",
        **KERNEL_PARAMS,
    )

    sampling = SamplingParams(**KERNEL_SAMPLING)
    lora_request = (
        LoRARequest("user_adapter", 1, str(adapter_dir.resolve())) if adapter_dir else None
    )

    formatted_prompts = []
    meta = []
    for rid, prompt, gt, category in _iter_examples(records):
        formatted_prompts.append(_format_prompt(tokenizer, prompt))
        meta.append((rid, prompt, gt, category))

    print(f"Generating {len(formatted_prompts)} examples (greedy, max_tokens=7680) ...")
    t0 = time.time()
    outs = llm.generate(formatted_prompts, sampling, lora_request=lora_request)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({len(outs) / elapsed:.2f} examples/s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    per_cat: dict[str, list[int]] = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        for (rid, prompt, gt, category), o in zip(meta, outs, strict=True):
            text = o.outputs[0].text if o.outputs else ""
            extracted = extract_final_answer(text)
            is_correct = verify(gt, extracted)
            correct += int(is_correct)
            bucket = per_cat.setdefault(category or "unknown", [0, 0])
            bucket[0] += int(is_correct)
            bucket[1] += 1
            fh.write(
                json.dumps(
                    {
                        "id": rid,
                        "prompt": prompt,
                        "ground_truth": gt,
                        "category": category,
                        "generation": text,
                        "extracted": extracted,
                        "is_correct": is_correct,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n = len(meta)
    summary = {
        "total": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "per_category": {k: {"correct": v[0], "total": v[1], "acc": v[0] / v[1] if v[1] else 0.0}
                         for k, v in sorted(per_cat.items())},
        "wall_time_s": elapsed,
        "out_path": str(out_path),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--base-model", type=str, default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--adapter", type=Path, default=None, help="Optional PEFT LoRA adapter dir")
    parser.add_argument("--val-jsonl", type=Path, required=True, help="JSONL with {id, prompt, ground_truth, category}")
    parser.add_argument("--out", type=Path, default=Path("outputs/eval.jsonl"))
    args = parser.parse_args()

    if not args.val_jsonl.is_file():
        print(f"ERROR: validation file not found: {args.val_jsonl}", file=sys.stderr)
        return 1

    records = _load_jsonl(args.val_jsonl)
    summary = run_eval(args.base_model, args.adapter, records, args.out)

    print()
    print("=" * 60)
    print(f"Accuracy: {summary['correct']}/{summary['total']} = {summary['accuracy']:.4f}")
    print("=" * 60)
    for cat, stats in summary["per_category"].items():
        print(f"  {cat:>20s}  {stats['correct']:>4d}/{stats['total']:<4d}  {stats['acc']:.4f}")
    print()
    print(f"Output: {summary['out_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
