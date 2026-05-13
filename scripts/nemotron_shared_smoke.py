from __future__ import annotations

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    model_id = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    burn_seconds = float(os.environ.get("BURN_SECONDS", "60"))
    seq_len = int(os.environ.get("SEQ_LEN", "1024"))
    batch = int(os.environ.get("BATCH", "1"))

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
    )
    model.eval()

    vocab = min(int(getattr(model.config, "vocab_size", 32000)), 50000)
    input_ids = torch.randint(0, vocab, (batch, seq_len), device="cuda:0", dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, device="cuda:0")

    # Warmup a little before timed burn.
    with torch.inference_mode():
        for _ in range(2):
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    t0 = time.perf_counter()
    iters = 0
    with torch.inference_mode():
        while time.perf_counter() - t0 < burn_seconds:
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            iters += 1
    elapsed = time.perf_counter() - t0

    mem = {}
    for idx in range(torch.cuda.device_count()):
        mem[f"gpu{idx}_max_mem_gb"] = round(torch.cuda.max_memory_allocated(idx) / (1024**3), 2)

    payload = {
        "ok": True,
        "gpu_count": torch.cuda.device_count(),
        "burn_seconds": burn_seconds,
        "elapsed_s": round(elapsed, 3),
        "iters": iters,
        "seq_len": seq_len,
        "batch": batch,
    }
    payload.update(mem)
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
