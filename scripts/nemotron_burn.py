from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model_id = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    burn_seconds = float(os.environ.get("BURN_SECONDS", "60"))
    seq_len = int(os.environ.get("SEQ_LEN", "1024"))
    batch = int(os.environ.get("BATCH", "1"))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

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
        device_map={"": local_rank},
        token=token,
    )
    model.eval()

    # Synthetic but model-real forward workload.
    vocab = min(int(getattr(model.config, "vocab_size", 32000)), 50000)
    x = torch.randint(0, vocab, (batch, seq_len), device=device, dtype=torch.long)
    attn = torch.ones_like(x, device=device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    iters = 0
    with torch.inference_mode():
        while time.perf_counter() - t0 < burn_seconds:
            _ = model(input_ids=x, attention_mask=attn, use_cache=False)
            iters += 1
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    msg = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "gpu": torch.cuda.get_device_name(device),
        "burn_seconds": burn_seconds,
        "elapsed_s": round(t1 - t0, 3),
        "iters": iters,
        "seq_len": seq_len,
        "batch": batch,
        "max_mem_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
        "ok": True,
    }
    print(json.dumps(msg), flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
