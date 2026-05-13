"""Load cleaned SFT data from local HF snapshot dirs or the HF Hub."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build OpenAI-style messages from one dataset row."""
    if row.get("messages"):
        msgs = row["messages"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        if not isinstance(msgs, list):
            raise ValueError(f"messages must be list, got {type(msgs)}")
        return msgs

    prompt = (
        row.get("prompt")
        or row.get("instruction")
        or row.get("question")
        or row.get("user")
    )
    completion = (
        row.get("completion")
        or row.get("response")
        or row.get("output")
        or row.get("assistant")
        or row.get("answer")
        or row.get("ground_truth")
    )
    if prompt is None or completion is None:
        keys = sorted(row.keys())
        raise ValueError(
            "Row has neither messages nor (prompt,completion)-style columns. "
            f"Keys present: {keys}"
        )

    user_content = str(prompt).strip()
    asst = str(completion).strip()

    thinking = row.get("thinking") or row.get("assistant_thinking")
    if isinstance(thinking, str) and thinking.strip():
        asst = f"<think>\n{thinking.strip()}\n</think>\n\n{asst}"

    out: list[dict[str, str]] = [{"role": "user", "content": user_content}]
    system = row.get("system") or row.get("system_prompt")
    if isinstance(system, str) and system.strip():
        out.insert(0, {"role": "system", "content": system.strip()})
    out.append({"role": "assistant", "content": asst})
    return out


def load_sft_raw(*, dataset_dir: Path | None, hub_id: str | None, hf_token: str | None) -> Dataset:
    """Load train split without formatting (columns vary)."""
    if hub_id:
        print(f"Loading dataset from Hub: {hub_id}", file=sys.stderr)
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if hf_token:
            kwargs["token"] = hf_token
        ds = load_dataset(hub_id, split="train", **kwargs)
        return ds

    if dataset_dir is None:
        raise ValueError("Either dataset_dir or hub_id is required")

    root = dataset_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {root}")

    parquet = sorted(root.rglob("*.parquet"))
    jsonl_all = sorted(root.rglob("*.jsonl"))
    json_files = sorted(root.rglob("*.json"))

    if parquet:
        paths = [str(p) for p in parquet]
        print(f"Loading {len(paths)} parquet shard(s) under {root}", file=sys.stderr)
        return load_dataset("parquet", data_files=paths, split="train")

    if jsonl_all:
        train_jsonl = root / "train.jsonl"
        if train_jsonl.is_file():
            print(f"Loading jsonl {train_jsonl}", file=sys.stderr)
            return load_dataset("json", data_files=str(train_jsonl), split="train")
        train_shards = sorted(root.glob("train-*.jsonl"))
        if train_shards:
            paths = [str(p) for p in train_shards]
            print(f"Loading {len(paths)} train shard(s) under {root}", file=sys.stderr)
            return load_dataset("json", data_files=paths, split="train")
        jsonl = [p for p in jsonl_all if p.parent == root]
        if len(jsonl) == 1:
            print(f"Loading jsonl {jsonl[0]}", file=sys.stderr)
            return load_dataset("json", data_files=str(jsonl[0]), split="train")
        jsonl_train = [
            p
            for p in jsonl_all
            if "val" not in p.name.lower()
            and "eval" not in p.name.lower()
            and "greedy" not in p.name.lower()
        ]
        if len(jsonl_train) == 1:
            print(f"Loading jsonl {jsonl_train[0]}", file=sys.stderr)
            return load_dataset("json", data_files=str(jsonl_train[0]), split="train")
        if len(jsonl_train) > 1:
            paths = [str(p) for p in sorted(jsonl_train)]
            print(f"Loading {len(paths)} jsonl file(s) under {root}", file=sys.stderr)
            return load_dataset("json", data_files=paths, split="train")
        paths = [str(p) for p in jsonl_all]
        print(f"Loading {len(paths)} jsonl file(s) under {root}", file=sys.stderr)
        return load_dataset("json", data_files=paths, split="train")

    if json_files:
        print(f"Loading json {json_files[0]}", file=sys.stderr)
        return load_dataset("json", data_files=str(json_files[0]), split="train")

    raise FileNotFoundError(
        f"No parquet/jsonl/json train files found under {root}. "
        "Run `python -m data.download --sft-only` or pass --dataset-name."
    )


def normalize_messages_dataset(ds: Dataset) -> Dataset:
    """Ensure each row has a messages list."""

    def _one(ex: dict[str, Any]) -> dict[str, Any]:
        return {"messages": row_to_messages(ex)}

    return ds.map(_one, remove_columns=ds.column_names)


def dataset_to_text(
    ds: Dataset,
    tokenizer,
    *,
    enable_thinking: bool = True,
) -> Dataset:
    """Apply chat template; TRL trains on the text column."""

    def _apply_chat_template(messages: list[dict[str, str]]) -> str:
        if enable_thinking:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=True,
                )
            except TypeError:
                pass
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _batched(batch: dict[str, Any]) -> dict[str, Any]:
        texts = [_apply_chat_template(m) for m in batch["messages"]]
        return {"text": texts}

    return ds.map(_batched, batched=True, remove_columns=["messages"])
