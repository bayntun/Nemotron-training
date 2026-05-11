from train.csv_cipher_length_hint import augment_prompt_for_cipher_length_hint, cipher_length_hint


def test_cipher_hint_synthetic() -> None:
    p = (
        "In Alice's Wonderland, a secret cipher rewrites text.\n"
        "Examples:\n"
        "ab -> cd\n"
        "wx -> yz\n"
        "Now determine the result for: pq\n"
    )
    h = cipher_length_hint(p)
    assert h is not None
    assert "2" in h  # 2 chars, 1 word
    out = augment_prompt_for_cipher_length_hint(p, "auto")
    assert "Text-cipher structural hint" in out


def test_cipher_skips_generic_arrow_transform() -> None:
    """Generic `->` puzzles without encrypt/cipher wording must not get text-cipher hints."""
    p = (
        "In Wonderland, a secret rule maps strings.\n"
        "aa -> bb\n"
        "cc -> dd\n"
        "Now determine the result for: ee\n"
    )
    assert cipher_length_hint(p) is None


def test_cipher_skips_binary_prompt() -> None:
    p = (
        "In Alice's Wonderland, a secret bit manipulation transforms 8-bit binary.\n"
        "10101010 -> 01010101\n"
        "11001100 -> 00110011\n"
        "Now determine the result for: 11110000\n"
    )
    assert cipher_length_hint(p) is None


def test_no_hint_without_two_pairs() -> None:
    p = "In Alice's Wonderland, one example only: ab -> cd\nNow determine the result for: pq\n"
    assert cipher_length_hint(p) is None


def test_none_passthrough() -> None:
    p = "No arrows here"
    assert augment_prompt_for_cipher_length_hint(p, "none") == p
