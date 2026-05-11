"""Shared prompt gates so cipher hints never fire on binary / generic arrow transforms."""
from __future__ import annotations

import re

_TEXT_CIPHER_LABEL = re.compile(
    r"\b(secret\s+encryption|encryption\b|encrypt\b|decrypt|decode|ciphers?\b|ciphertext|plaintext)\b",
    re.IGNORECASE,
)


def is_binary_style_prompt(prompt_lower: str) -> bool:
    return any(
        k in prompt_lower
        for k in (
            "binary",
            "8-bit",
            "8 bit",
            "bit manipulation",
            "bit-manipulation",
        )
    )


def is_text_cipher_labeled_prompt(prompt: str) -> bool:
    """True only for explicit text encrypt/decrypt/cipher wording; excludes binary puzzles."""
    pl = prompt.lower()
    if is_binary_style_prompt(pl):
        return False
    return bool(_TEXT_CIPHER_LABEL.search(prompt))
