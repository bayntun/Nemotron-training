from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    use_ddp = world_size > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank if use_ddp else 0)

    import causal_conv1d  # noqa: F401
    import mamba_ssm  # noqa: F401

    # Tunables for smoke vs burn runs.
    steps = int(os.environ.get("SMOKE_STEPS", "12"))
    burn_seconds = float(os.environ.get("BURN_SECONDS", "0"))
    reserve_gb = float(os.environ.get("RESERVE_GB", "0"))
    mat_dim = int(os.environ.get("MAT_DIM", "4096"))

    reserve = None
    if reserve_gb > 0:
        reserve_numel = int((reserve_gb * (1024**3)) // 2)  # fp16 => 2 bytes
        reserve = torch.empty(reserve_numel, device=device, dtype=torch.float16)
        reserve.fill_(1.0)
    x = torch.randn(mat_dim, mat_dim, device=device, dtype=torch.float16)
    w = torch.randn(mat_dim, mat_dim, device=device, dtype=torch.float16)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    if burn_seconds > 0:
        end_t = time.perf_counter() + burn_seconds
        iters = 0
        while time.perf_counter() < end_t:
            x = torch.matmul(x, w)
            x = torch.relu(x)
            iters += 1
    else:
        iters = 0
        for _ in range(steps):
            x = torch.matmul(x, w)
            x = torch.relu(x)
            iters += 1
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    msg = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "gpu": torch.cuda.get_device_name(device),
        "elapsed_s": round(t1 - t0, 3),
        "reserve_gb": reserve_gb,
        "mat_dim": mat_dim,
        "burn_seconds": burn_seconds,
        "iters": iters,
        "max_mem_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
        "ok": True,
    }
    print(json.dumps(msg), flush=True)

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
