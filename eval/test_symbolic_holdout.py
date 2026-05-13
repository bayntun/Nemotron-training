"""Tests for symbolic-equation holdout prompt filter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.symbolic_holdout import is_symbolic_equation_holdout_prompt

REPO = Path(__file__).resolve().parent.parent
VAL = REPO / "data" / "cache" / "nemotron_sft_deepseek" / "val_greedy.jsonl"

EXPECTED_SYMBOLIC_IDS = {
    "fbd5fe63",
    "faf1121c",
    "f8731f21",
    "f3e08a24",
    "f9799a68",
    "fb623471",
    "fc3982df",
    "f45dd620",
    "fc759a1a",
}


@pytest.mark.skipif(not VAL.is_file(), reason="val_greedy.jsonl not present")
def test_val_greedy_symbolic_ids_match_heuristic() -> None:
    sym: set[str] = set()
    non_sym_with_false_positive: list[str] = []
    with VAL.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            pid = str(rec.get("id", ""))
            p = str(rec.get("prompt") or "")
            if is_symbolic_equation_holdout_prompt(p):
                sym.add(pid)
            elif pid in EXPECTED_SYMBOLIC_IDS:
                non_sym_with_false_positive.append(pid)
    assert non_sym_with_false_positive == []
    assert sym == EXPECTED_SYMBOLIC_IDS
