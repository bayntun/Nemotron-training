import json
import subprocess
import time
from pathlib import Path

import torch
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


def run_case(case: dict[str, int]) -> dict:
    device = torch.device("cuda", 0)
    dtype = torch.float16
    try:
        model = TinyTrainModel(hidden=case["hidden"], layers=case["layers"]).to(device=device, dtype=dtype)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(case["batch"], case["seq"], case["hidden"], device=device, dtype=dtype)
        target = torch.randn(case["batch"], case["seq"], case["hidden"], device=device, dtype=dtype)

        def step() -> float:
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = ((out - target) ** 2).mean()
            loss.backward()
            opt.step()
            torch.cuda.synchronize(device)
            return time.perf_counter() - t0

        for _ in range(4):
            _ = step()
        times = [step() for _ in range(16)]
        avg_s = sum(times) / len(times)
        toks_per_sec = (case["batch"] * case["seq"]) / avg_s

        nvsmi = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        util = mem = power = None
        if nvsmi.returncode == 0 and nvsmi.stdout.strip():
            parts = [x.strip() for x in nvsmi.stdout.strip().split(",")]
            if len(parts) >= 3:
                util, mem, power = float(parts[0]), float(parts[1]), float(parts[2])

        return {
            "case": case,
            "status": "ok",
            "avg_step_s": round(avg_s, 4),
            "tokens_per_sec": round(toks_per_sec, 2),
            "gpu_util_snapshot": util,
            "gpu_mem_mib_snapshot": mem,
            "gpu_power_w_snapshot": power,
        }
    except Exception as e:
        return {"case": case, "status": "fail", "error": str(e)}


def main() -> None:
    out_dir = Path("/kaggle/working/rtx6000_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["nvidia-smi"], check=False)
    cases = [
        {"name": "s1", "batch": 8, "seq": 512, "hidden": 3072, "layers": 10},
        {"name": "s2", "batch": 8, "seq": 1024, "hidden": 3072, "layers": 10},
        {"name": "s3", "batch": 12, "seq": 1024, "hidden": 3072, "layers": 10},
        {"name": "s4", "batch": 8, "seq": 1536, "hidden": 3072, "layers": 10},
        {"name": "s5", "batch": 12, "seq": 1536, "hidden": 3072, "layers": 10},
        {"name": "s6", "batch": 8, "seq": 1024, "hidden": 4096, "layers": 12},
        {"name": "s7", "batch": 16, "seq": 1536, "hidden": 3072, "layers": 10},
        {"name": "s8", "batch": 8, "seq": 2048, "hidden": 3072, "layers": 10},
    ]
    results = [run_case(c) for c in cases]
    ok = [r for r in results if r.get("status") == "ok"]
    best_tps = max(ok, key=lambda x: x["tokens_per_sec"]) if ok else None
    best_mem = max(ok, key=lambda x: x["gpu_mem_mib_snapshot"] or -1) if ok else None
    payload = {"results": results, "best_tokens_per_sec": best_tps, "best_memory_snapshot": best_mem}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
