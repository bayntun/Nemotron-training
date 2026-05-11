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
    assert "Structural hint" in out


def test_no_hint_without_two_pairs() -> None:
    p = "In Alice's Wonderland, one example only: ab -> cd\nNow determine the result for: pq\n"
    assert cipher_length_hint(p) is None


def test_none_passthrough() -> None:
    p = "No arrows here"
    assert augment_prompt_for_cipher_length_hint(p, "none") == p
