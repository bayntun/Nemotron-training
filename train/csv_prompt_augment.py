"""Chain all optional CSV User-prompt augmentations (train + eval)."""
from __future__ import annotations

from train.csv_cipher_length_hint import augment_prompt_for_cipher_length_hint
from train.csv_encrypt_lexical_hint import augment_prompt_for_encrypt_lexical_hint
from train.csv_equation_shape_hint import augment_prompt_for_equation_shape_hint
from train.csv_numeric_baseline import augment_prompt_for_numeric_baseline


def augment_csv_user_prompt(
    raw_prompt: str,
    *,
    numeric: str,
    cipher: str,
    encrypt: str,
    equation: str,
) -> str:
    p = raw_prompt
    p = augment_prompt_for_numeric_baseline(p, numeric)  # type: ignore[arg-type]
    p = augment_prompt_for_cipher_length_hint(p, cipher)  # type: ignore[arg-type]
    p = augment_prompt_for_encrypt_lexical_hint(p, encrypt)  # type: ignore[arg-type]
    p = augment_prompt_for_equation_shape_hint(p, equation)  # type: ignore[arg-type]
    return p
