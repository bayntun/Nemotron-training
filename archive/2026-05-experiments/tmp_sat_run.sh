#!/usr/bin/env bash
set -euo pipefail

LOG=/tmp/v100_sat_log.csv
rm -f "$LOG"

(
  for _ in $(seq 1 80); do
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits >> "$LOG"
    sleep 1
  done
) &
MONPID=$!

docker exec jupyter-abayntun bash -lc \
  "cd /home/jovyan/work/Nemotron-training && torchrun --nproc_per_node=4 scripts/benchmark_gpu.py --batch 8 --seq 512 --hidden 3072 --layers 10 --warmup 5 --steps 30 --dtype fp16"

wait "$MONPID"

python3 - <<'PY'
import pathlib, statistics

p = pathlib.Path("/tmp/v100_sat_log.csv")
rows = []
for line in p.read_text().splitlines():
    parts = [x.strip() for x in line.split(",")]
    if len(parts) != 3:
        continue
    try:
        rows.append(tuple(float(x) for x in parts))
    except ValueError:
        pass

if not rows:
    print("no_samples")
    raise SystemExit(0)

util = [r[0] for r in rows]
mem = [r[1] for r in rows]
pwr = [r[2] for r in rows]
util_sorted = sorted(util)
idx = max(0, int(0.95 * len(util_sorted)) - 1)

print(f"samples={len(rows)}")
print(f"util_avg={statistics.mean(util):.1f} util_p95={util_sorted[idx]:.1f} util_max={max(util):.1f}")
print(f"mem_avg_mib={statistics.mean(mem):.0f} mem_max_mib={max(mem):.0f}")
print(f"pwr_avg_w={statistics.mean(pwr):.1f} pwr_max_w={max(pwr):.1f}")
PY
