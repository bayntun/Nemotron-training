import json
import os
import re
import subprocess
import time
from pathlib import Path


REPO = Path("/home/jovyan/work/Nemotron-training")
SCRIPT = REPO / "scripts/benchmark_gpu.py"
OUT = REPO / "outputs/benchmark_sweep_v100.json"

CASES = [
    {"batch": 4, "seq": 256, "hidden": 3072, "layers": 10},
    {"batch": 8, "seq": 256, "hidden": 3072, "layers": 10},
    {"batch": 8, "seq": 384, "hidden": 3072, "layers": 10},
    {"batch": 8, "seq": 512, "hidden": 3072, "layers": 10},
    {"batch": 4, "seq": 512, "hidden": 4096, "layers": 12},
]


def parse_metrics(text: str) -> dict[str, float]:
    def grab(name: str) -> float:
        m = re.search(rf"{name}=([0-9.]+)", text)
        if not m:
            raise ValueError(f"missing metric {name}")
        return float(m.group(1))

    return {
        "avg_step_s": grab("avg_step_s"),
        "tokens_per_sec": grab("tokens_per_sec"),
        "approx_tflops": grab("approx_tflops"),
    }


def run_case(case: dict[str, int]) -> dict:
    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        str(SCRIPT),
        "--batch",
        str(case["batch"]),
        "--seq",
        str(case["seq"]),
        "--hidden",
        str(case["hidden"]),
        "--layers",
        str(case["layers"]),
        "--warmup",
        "3",
        "--steps",
        "12",
        "--dtype",
        "fp16",
    ]
    env = os.environ.copy()
    env.setdefault("TORCH_DISTRIBUTED_USE_LIBUV", "0")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    record = {"case": case, "elapsed_s": round(elapsed, 2), "returncode": p.returncode}
    if p.returncode == 0:
        metrics = parse_metrics(p.stdout)
        record.update(metrics)
        record["status"] = "ok"
    else:
        record["status"] = "fail"
        record["stderr_tail"] = "\n".join(p.stderr.splitlines()[-20:])
    return record


def main() -> int:
    results = [run_case(c) for c in CASES]
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        best = max(ok, key=lambda x: x["tokens_per_sec"])
    else:
        best = None
    payload = {"results": results, "best": best}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"saved={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
