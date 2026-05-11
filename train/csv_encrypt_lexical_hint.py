"""
Lexical + word-count hints for encrypt/decrypt few-shot rows (Wonderland text ciphers).

Uses only the prompt: collects plaintext tokens from RHS of `->` examples, checks
per-line word-count alignment, and parses the final ciphertext query when possible.
"""
from __future__ import annotations

import re
from typing import Literal

from train.csv_hint_gates import is_binary_style_prompt, is_text_cipher_labeled_prompt

EncryptStrategy = Literal["none", "auto"]

_STOP = frozenset(
    {
        "a",
        "an",
        "as",
        "at",
        "by",
        "for",
        "in",
        "of",
        "on",
        "or",
        "to",
    }
)

_QUERY_DECRYPT = re.compile(
    r"(?:decrypt|decode)\s+the\s+following\s+text\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_QUERY_FALLBACK = re.compile(
    r"(?:following\s+text|following\s+ciphertext)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _line_arrow_pairs(prompt: str) -> list[tuple[str, str]]:
    """One `lhs -> rhs` per line (matches Wonderland encrypt prompts)."""
    out: list[tuple[str, str]] = []
    for line in prompt.splitlines():
        if "->" not in line and "→" not in line:
            continue
        sep = "->" if "->" in line else "→"
        if line.count(sep) != 1:
            continue
        left, right = line.split(sep, 1)
        left, right = left.strip(), right.strip()
        if len(left) < 2 or len(right) < 2:
            continue
        low = left.lower()
        if low.startswith("here ") or low.startswith("now,") or "example" in low and ":" in left[:20]:
            continue
        out.append((left, right))
    return out


def _extract_ciphertext_query(prompt: str) -> str | None:
    for pat in (_QUERY_DECRYPT, _QUERY_FALLBACK):
        m = pat.search(prompt)
        if m:
            return m.group(1).strip()
    return None


def encrypt_lexical_hint(prompt: str) -> str | None:
    pl = prompt.lower()
    if is_binary_style_prompt(pl):
        return None
    if not is_text_cipher_labeled_prompt(prompt):
        return None
    if "->" not in prompt and "→" not in prompt:
        return None
    pairs = _line_arrow_pairs(prompt)
    if len(pairs) < 2:
        return None

    wc_in: list[int] = []
    vocab: list[str] = []
    for left, right in pairs:
        lw = left.split()
        rw = right.split()
        wc_in.append(len(lw))
        for w in rw:
            w2 = w.strip()
            if not w2:
                continue
            if w2.lower() in _STOP and len(w2) <= 2:
                continue
            vocab.append(w2)

    if not vocab:
        return None

    uniform_w = len(set(wc_in)) == 1
    w_ex = wc_in[0] if uniform_w else None

    q = _extract_ciphertext_query(prompt)
    w_q = len(q.split()) if q else None

    uniq = sorted(set(vocab), key=lambda s: (s.lower(), s))[:28]
    lex = ", ".join(uniq)

    parts = [
        "Decrypt / cipher lexical hint: plaintext in the examples reuses a small Wonderland vocabulary.",
        f"Frequent plaintext tokens (hints only): {lex}.",
    ]
    if uniform_w and w_ex is not None:
        parts.append(f"Each example line has **{w_ex}** ciphertext words mapped to **{w_ex}** plaintext words.")
    if q and w_q is not None:
        parts.append(f"The query ciphertext has **{w_q}** words; answer with **{w_q}** plaintext words on one line if word boundaries are preserved like the examples.")
    parts.append("If a token is not in the list, infer it from the cipher pattern shown in the examples.")

    return " ".join(parts)


def augment_prompt_for_encrypt_lexical_hint(prompt: str, strategy: EncryptStrategy) -> str:
    if strategy == "none":
        return prompt
    hint = encrypt_lexical_hint(prompt)
    if hint is None:
        return prompt
    return f"{prompt.rstrip()}\n\n{hint}"
