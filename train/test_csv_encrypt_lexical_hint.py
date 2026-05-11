from train.csv_encrypt_lexical_hint import augment_prompt_for_encrypt_lexical_hint, encrypt_lexical_hint


def test_encrypt_hint_multiline() -> None:
    p = """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
aa bb -> alice studies
cc dd -> the dark king
Now, decrypt the following text: ee ff
"""
    h = encrypt_lexical_hint(p)
    assert h is not None
    assert "alice" in h.lower() or "studies" in h.lower()
    out = augment_prompt_for_encrypt_lexical_hint(p, "auto")
    assert "Lexical hint" in out or "lexical hint" in out


def test_encrypt_none_without_arrows() -> None:
    p = "In Alice's Wonderland, decrypt the message with no arrows"
    assert encrypt_lexical_hint(p) is None
