import csv
import json
from pathlib import Path


def main() -> None:
    src = Path("/home/jovyan/work/train.csv")
    out_dir = Path("/home/jovyan/work/Nemotron-training/data/cache/sft")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "train_from_csv.jsonl"
    rows = 0
    with src.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
        for r in csv.DictReader(f):
            prompt = (r.get("prompt") or "").strip()
            answer = str(r.get("answer") or "").strip()
            if not prompt or not answer:
                continue
            w.write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False) + "\n")
            rows += 1
    print(f"rows_written={rows}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
