#!/usr/bin/env python3
"""
Greedy-ish eval for Nemotron + PEFT on a val JSONL **without vLLM** (vLLM may reject Nemotron-H).

Loads 4bit base + LoRA adapter, formats prompts like ``eval.greedy_harness`` (thinking on),
runs ``model.generate``, scores with ``eval.grader`` (same boxed extraction as the competition).

Run on the Linux GPU box from repo root::

    python3 scripts/eval_nemotron_holdout_transformers.py \\
      --adapter outputs/nemotron_sft_deepseek_forward_v2 \\
      --val-jsonl data/cache/nemotron_sft_deepseek/val_greedy.jsonl \\
      --out-jsonl outputs/nemotron_sft_deepseek_forward_v2/eval_holdout.jsonl

Symbolic equation holdout only (smaller run)::

    python3 scripts/eval_nemotron_holdout_transformers.py ... \\
      --subset symbolic-equations

After a full eval JSONL exists, summarize symbolic rows only::

    python3 scripts/report_symbolic_holdout_eval.py \\
      --eval-jsonl outputs/nemotron_sft_deepseek_forward_v2/eval_holdout.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.grader import extract_final_answer, verify
from eval.symbolic_holdout import is_symbolic_equation_holdout_prompt


def _format_prompt(tokenizer, user_content: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-name-or-path",
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    )
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows in val file")
    ap.add_argument(
        "--subset",
        choices=("all", "symbolic-equations"),
        default="all",
        help="symbolic-equations = equation/symbol transform holdouts only.",
    )
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")
        load_dotenv(REPO / "bootstrap" / "secrets_local.env")
    except ImportError:
        pass

    import gc
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from train.sft import _force_nemotron_mamba_torch_forward_path, _patch_nemotron_h_moe_index_add_dtype

    if not args.val_jsonl.is_file():
        print(f"ERROR: missing {args.val_jsonl}", file=sys.stderr)
        return 2
    if not (args.adapter / "adapter_config.json").is_file():
        print(f"ERROR: not an adapter dir: {args.adapter}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    with args.val_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if args.subset == "symbolic-equations":
        rows = [r for r in rows if is_symbolic_equation_holdout_prompt(str(r.get("prompt") or ""))]
    if args.max_rows:
        rows = rows[: args.max_rows]

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading base model (4bit) …", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    _force_nemotron_mamba_torch_forward_path(base)
    _patch_nemotron_h_moe_index_add_dtype(base)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading adapter {args.adapter} …", flush=True)
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    model.eval()
    dev = next(model.parameters()).device

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    n_done = 0
    results: list[dict] = []
    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for i, rec in enumerate(rows):
            rid = str(rec.get("id", i))
            prompt = str(rec.get("prompt") or "")
            gt = str(rec.get("ground_truth") or rec.get("answer") or "").strip()
            category = str(rec.get("category", ""))
            if not prompt or not gt:
                continue
            n_done += 1
            text_in = _format_prompt(tokenizer, prompt)
            enc = tokenizer(text_in, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.inference_mode():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = tokenizer.decode(
                out_ids[0][enc["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            extracted = extract_final_answer(gen)
            ok = verify(gt, extracted)
            correct += int(ok)
            rec_out = {
                "id": rid,
                "prompt": prompt,
                "ground_truth": gt,
                "category": category,
                "generation": gen,
                "extracted": extracted,
                "is_correct": ok,
            }
            results.append(rec_out)
            out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {"i": n_done, "n": len(rows), "id": rid, "ok": ok, "gt": gt, "extracted": extracted},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    n = n_done
    acc = correct / n if n else 0.0
    print(f"\nAccuracy: {correct}/{n} = {acc:.4f}", flush=True)
    print(f"Wrote {args.out_jsonl}", flush=True)

    sym_results = [r for r in results if is_symbolic_equation_holdout_prompt(str(r.get("prompt") or ""))]
    if sym_results and args.subset == "all":
        sc = sum(1 for r in sym_results if r.get("is_correct"))
        sn = len(sym_results)
        print(f"\nSymbolic equation subset (same heuristic): {sc}/{sn} = {sc / sn:.4f}", flush=True)
        for r in sym_results:
            ok = bool(r.get("is_correct"))
            tag = "OK" if ok else "XX"
            print(
                f"  [{tag}] id={r.get('id')} gt={r.get('ground_truth')!r} extracted={r.get('extracted')!r}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
