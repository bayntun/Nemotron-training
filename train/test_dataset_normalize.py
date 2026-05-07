"""Tests for SFT row → messages normalization."""

from __future__ import annotations

from train._dataset import row_to_messages


def test_messages_pass_through():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert row_to_messages({"messages": msgs}) == msgs


def test_prompt_completion_pair():
    m = row_to_messages({"prompt": "Question?", "completion": "Answer."})
    assert m[0]["role"] == "user"
    assert m[0]["content"] == "Question?"
    assert m[-1]["role"] == "assistant"
    assert m[-1]["content"] == "Answer."


def test_system_prompt():
    m = row_to_messages(
        {"system": "You are helpful.", "prompt": "Hi", "response": "Hello"}
    )
    assert m[0]["role"] == "system"
    assert m[1]["role"] == "user"
