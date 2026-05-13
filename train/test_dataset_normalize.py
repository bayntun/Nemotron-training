"""Tests for SFT row → messages normalization."""

from __future__ import annotations

from pathlib import Path

from train._dataset import row_to_messages, load_sft_raw


def test_messages_pass_through():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert row_to_messages({"messages": msgs}) == msgs


def test_prompt_completion_pair():
    m = row_to_messages({"prompt": "Question?", "completion": "Answer."})
    assert m[0]["role"] == "user"
    assert m[0]["content"] == "Question?"
    assert m[-1]["role"] == "assistant"
    assert m[-1]["content"] == "Answer."


def test_load_sft_raw_prefers_train_jsonl_when_val_also_present(tmp_path: Path):
    (tmp_path / "train.jsonl").write_text(
        '{"prompt":"p1","answer":"a1"}\n{"prompt":"p2","answer":"a2"}\n',
        encoding="utf-8",
    )
    (tmp_path / "val_greedy.jsonl").write_text(
        '{"id":"x","prompt":"q","ground_truth":"g","category":"c"}\n',
        encoding="utf-8",
    )
    ds = load_sft_raw(dataset_dir=tmp_path, hub_id=None, hf_token=None)
    assert len(ds) == 2
    assert "prompt" in ds.column_names and "answer" in ds.column_names


def test_system_prompt():
    m = row_to_messages(
        {"system": "You are helpful.", "prompt": "Hi", "response": "Hello"}
    )
    assert m[0]["role"] == "system"
    assert m[1]["role"] == "user"
