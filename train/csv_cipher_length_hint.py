"""
Optional hints for string / cipher-style Wonderland few-shot rows.

If the prompt contains multiple `left -> right` examples where every pair preserves
both character length and whitespace-delimited word count, we infer the same
constraint applies to the final query string parsed from the prompt.

This is structural only (no label leakage). The true cipher may still violate
length on edge cases; the hint says to trust the examples if they conflict.
"""
from __future__ import annotations

import re
from typing import Literal

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
        f"Structural hint: each listed example maps a string to another with the **same** "
        f"character count and **same** word count as the input. "
        f"The query string has **{n_chars}** characters and **{n_words}** word(s); "
        f"your answer should match those counts on a single line unless the examples clearly do otherwise."
    )


def augment_prompt_for_cipher_length_hint(prompt: str, strategy: CipherStrategy) -> str:
    if strategy == "none":
        return prompt
    hint = cipher_length_hint(prompt)
    if hint is None:
        return prompt
    return f"{prompt.rstrip()}\n\n{hint}"
