from train.csv_hint_gates import is_binary_style_prompt, is_text_cipher_labeled_prompt


def test_cipher_label_requires_vocab() -> None:
    assert not is_text_cipher_labeled_prompt(
        "Wonderland rule:\naa -> bb\ncc -> dd\nNow: ee\n"
    )
    assert is_text_cipher_labeled_prompt(
        "Wonderland, secret encryption on text:\naa -> bb\n"
    )


def test_binary_excluded() -> None:
    pl = "8-bit binary manipulation decode output"
    assert is_binary_style_prompt(pl.lower())
    assert not is_text_cipher_labeled_prompt(pl)
