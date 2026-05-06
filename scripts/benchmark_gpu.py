from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Linear(4 * hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TinyTrainModel(nn.Module):
    def __init__(self, hidden: int, layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([Block(hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic train-step GPU benchmark")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()


def setup_ddp() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return use_ddp, world_size, rank, local_rank


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    use_ddp, world_size, rank, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank if use_ddp else 0)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        if rank == 0:
            print("bf16 unsupported on this GPU, falling back to fp16")
        dtype = torch.float16

    model = TinyTrainModel(hidden=args.hidden, layers=args.layers).to(device=device, dtype=dtype)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    x = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)
    target = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)

    def step() -> float:
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        return t1 - t0

    for _ in range(args.warmup):
        _ = step()

    times = [step() for _ in range(args.steps)]
    avg_s = sum(times) / len(times)

    global_tokens_per_step = args.batch * args.seq * world_size
    toks_per_sec = global_tokens_per_step / avg_s

    # Rough compute proxy (not exact FLOP accounting)
    approx_flops_per_token = 2.0 * args.layers * args.hidden * (4 * args.hidden) * 2.0
    approx_tflops = (toks_per_sec * approx_flops_per_token) / 1e12

    if rank == 0:
        gpu_name = torch.cuda.get_device_name(device)
        print("=== benchmark_result ===")
        print(f"gpu={gpu_name}")
        print(f"world_size={world_size}")
        print(f"dtype={str(dtype).replace('torch.', '')}")
        print(
            f"batch={args.batch} seq={args.seq} hidden={args.hidden} layers={args.layers}"
        )
        print(f"avg_step_s={avg_s:.6f}")
        print(f"tokens_per_sec={toks_per_sec:.2f}")
        print(f"approx_tflops={approx_tflops:.2f}")

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
