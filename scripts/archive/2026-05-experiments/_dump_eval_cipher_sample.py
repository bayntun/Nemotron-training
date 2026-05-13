"""One-off: print a sample row from eval_details.jsonl (run inside container or locally)."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/jovyan/work/Nemotron-training/outputs/csv_train_ddp_v5_cipher_hints/eval_details.jsonl"
with open(path, encoding="utf-8") as f:
    rows = [json.loads(l) for l in f if l.strip()]

# 1) Multi-word gold with "->" in stored prompt (hint path)
for r in rows:
    p = r.get("prompt", "")
    gt = str(r.get("gt", ""))
    if " " in gt.strip() and ("->" in p or "→" in p):
        print("=== Sample: multi-word + arrow in prompt (likely cipher family) ===\n")
        for k in ("id", "ok_full", "gt", "pred", "gen_full"):
            print(f"{k}: {r.get(k)!r}")
        print("\n--- full User prompt (as stored; may include hints) ---\n")
        print(p)
        sys.exit(0)

# 2) Any multi-word
for r in rows:
    gt = str(r.get("gt", ""))
    if " " in gt.strip():
        print("=== Sample: multi-word phrase (no arrow in prompt) ===\n")
        for k in ("id", "ok_full", "gt", "pred", "gen_full"):
            print(f"{k}: {r.get(k)!r}")
        print("\n--- prompt excerpt ---\n")
        print((r.get("prompt") or "")[:3500])
        sys.exit(0)

print("No suitable row found")
