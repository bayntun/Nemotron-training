import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.grader import extract_final_answer, verify

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = Path("/home/jovyan/work/Nemotron-training/outputs/sft_smoke_gpu")
DATA = Path("/home/jovyan/work/Nemotron-training/tmp/sft_smoke/train.jsonl")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, str(ADAPTER))
model.eval()

rows = [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines() if x.strip()]

correct = 0
for r in rows:
    prompt = r["prompt"]
    gt = str(r["completion"])
    messages = [{"role": "user", "content": prompt}]
    inp = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    inp = inp.to(model.device)
    out = model.generate(inp, max_new_tokens=24, temperature=0.0, do_sample=False)
    gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
    extracted = extract_final_answer(gen)
    ok = bool(verify(gt, extracted))
    correct += int(ok)
    print(f"prompt={prompt!r} gt={gt!r} gen={gen!r} extracted={extracted!r} ok={ok}")

print(f"accuracy={correct}/{len(rows)}={correct/len(rows):.3f}")
