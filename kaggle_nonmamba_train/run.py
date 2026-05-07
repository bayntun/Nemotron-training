import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel


def find_train_csv() -> Path:
    matches = sorted(Path("/kaggle/input").glob("**/train.csv"))
    if not matches:
        raise FileNotFoundError("train.csv not found under /kaggle/input")
    return matches[0]


def build_vocab(texts: list[str], max_vocab: int = 12000) -> dict[str, int]:
    freq = {}
    for t in texts:
        for tok in t.split():
            freq[tok] = freq.get(tok, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[: max_vocab - 2]
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, (tok, _) in enumerate(items, start=2):
        vocab[tok] = i
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> torch.Tensor:
    ids = [vocab.get(t, 1) for t in text.split()][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def train_eval(rows: list[dict[str, str]], cfg: dict, vocab: dict[str, int], device: torch.device) -> dict:
    eval_size = 120
    train_rows = rows[:1200]
    holdout = train_rows[-eval_size:]
    train_rows = train_rows[:-eval_size]

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=cfg["max_length"],
            n_ctx=cfg["max_length"],
            n_embd=768,
            n_layer=8,
            n_head=12,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            pad_token_id=0,
        )
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    train_tensors = [
        encode(f"User: {r['prompt']} Assistant: {r['answer']}", vocab, cfg["max_length"]) for r in train_rows
    ]
    random.shuffle(train_tensors)

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    start = time.time()
    steps = 0
    seen_tokens = 0
    bs = cfg["batch"]
    ga = cfg["grad_accum"]
    epochs = int(cfg["epochs"] * 100)
    repeat = max(1, epochs // 50)
    for _ in range(repeat):
        for i in range(0, len(train_tensors), bs):
            batch = torch.stack(train_tensors[i : i + bs]).to(device)
            labels = batch.clone()
            out = model(input_ids=batch, labels=labels)
            (out.loss / ga).backward()
            steps += 1
            seen_tokens += batch.numel()
            if steps % ga == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
    elapsed = max(1e-6, time.time() - start)
    tok_per_sec = int(seen_tokens / elapsed)
    max_mem_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2)

    model.eval()
    with torch.no_grad():
        losses = []
        for r in holdout:
            x = encode(f"User: {r['prompt']} Assistant: {r['answer']}", vocab, cfg["max_length"]).unsqueeze(0).to(device)
            losses.append(model(input_ids=x, labels=x).loss.item())
    ppl = round(float(torch.exp(torch.tensor(sum(losses) / len(losses))).item()), 3)

    return {
        "name": cfg["name"],
        "perplexity": ppl,
        "eval_rows": len(holdout),
        "tokens_per_sec": tok_per_sec,
        "max_memory_gb": max_mem_gb,
        "config": cfg,
    }


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)

    print("=== env ===", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)} cuda={torch.version.cuda} torch={torch.__version__}", flush=True)

    train_csv = find_train_csv()
    rows = list(csv.DictReader(train_csv.open("r", encoding="utf-8")))
    rows = [r for r in rows if "prompt" in r and "answer" in r]
    texts = [f"User: {r['prompt']} Assistant: {r['answer']}" for r in rows[:1500]]
    vocab = build_vocab(texts)
    print(f"train_csv={train_csv} rows={len(rows)} vocab={len(vocab)}", flush=True)
    device = torch.device("cuda")

    sweep = [
        {"name": "a", "max_length": 384, "batch": 1, "grad_accum": 4, "epochs": 0.5},
        {"name": "b", "max_length": 512, "batch": 1, "grad_accum": 4, "epochs": 0.5},
        {"name": "c", "max_length": 512, "batch": 2, "grad_accum": 2, "epochs": 0.5},
    ]

    results = []
    for cfg in sweep:
        print(f"=== run {cfg['name']} ===", flush=True)
        results.append(train_eval(rows, cfg, vocab, device))
        print(json.dumps(results[-1]), flush=True)

    best = min(results, key=lambda x: x["perplexity"])
    summary = {"results": results, "best": best}
    out = Path("/kaggle/working/nonmamba_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
