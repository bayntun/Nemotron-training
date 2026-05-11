"""
Length-preserving hints only for **explicitly labeled text cipher / encryption** prompts.

Generic `A -> B` transformations (binary puzzles, equation-adjacent patterns, etc.)
must NOT receive this hint — those families differ from Wonderland encrypt/decrypt copy.
"""
from __future__ import annotations

import re
from typing import Literal

from train.csv_hint_gates import is_text_cipher_labeled_prompt

CipherStrategy = Literal["none", "auto"]

# Capture "A -> B" where neither side is huge; avoid swallowing prose.
_ARROW = re.compile(
    r"([^\n]{1,160}?)\s*(?:->|→)\s*([^\n]{1,160}?)(?=\s*(?:\n|,|\s{2,}|Here|Now|Apply|Determine|\Z))",
    re.IGNORECASE,
)

_QUERY_PATTERNS = (
    re.compile(r"determine\s+the\s+result\s+for\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(
        r"(?:transform|encode|decode|apply)\s+the\s+following[^:]{0,40}:\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"following\s+(?:text|string|message|phrase|input)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"now,?\s+[^:]{0,80}:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
)


def _strip_wrapping(s: str) -> str:
    s = s.strip()
    for ch in ('"', "'", "`"):
        if len(s) >= 2 and s[0] == ch and s[-1] == ch:
            s = s[1:-1].strip()
    return s


def _word_count(s: str) -> int:
    return len(s.split()) if s.strip() else 0


def _length_preserving_pairs(prompt: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for m in _ARROW.finditer(prompt):
        left, right = _strip_wrapping(m.group(1)), _strip_wrapping(m.group(2))
        if not left or not right:
            continue
        if "->" in left or "→" in left or "->" in right or "→" in right:
            continue
        if len(left) != len(right):
            continue
        if _word_count(left) != _word_count(right):
            continue
        pairs.append((left, right))
    return pairs


def _extract_query_string(prompt: str) -> str | None:
    for pat in _QUERY_PATTERNS:
        m = pat.search(prompt)
        if m:
            return _strip_wrapping(m.group(1))
    return None


def cipher_length_hint(prompt: str) -> str | None:
    if not is_text_cipher_labeled_prompt(prompt):
        return None
    pairs = _length_preserving_pairs(prompt)
    if len(pairs) < 2:
        return None
    q = _extract_query_string(prompt)
    if not q:
        return None
    n_chars = len(q)
    n_words = _word_count(q)
    if n_chars == 0:
        return None
    return (
        f"Text-cipher structural hint: each listed ciphertext→plaintext example preserves "
        f"**same** character count and **same** word count. "
        f"The query ciphertext has **{n_chars}** characters and **{n_words}** word(s); "
        f"match those counts on one plaintext line unless the examples clearly do otherwise."
    )


def augment_prompt_for_cipher_length_hint(prompt: str, strategy: CipherStrategy) -> str:
    if strategy == "none":
        return prompt
    hint = cipher_length_hint(prompt)
    if hint is None:
        return prompt
    return f"{prompt.rstrip()}\n\n{hint}"
