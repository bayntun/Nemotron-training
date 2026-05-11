from train.csv_binary_arrow_hint import augment_prompt_for_binary_arrow_hint, binary_arrow_hint

SAMPLE = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
01111001 -> 11110010
01100100 -> 11001000

Now, determine the output for: 00100101
"""


def test_binary_hint_present() -> None:
    h = binary_arrow_hint(SAMPLE)
    assert h is not None
    assert "8" in h
    assert "Binary arrow-rule hint" in h
    assert "Binary arrow-rule hint" in augment_prompt_for_binary_arrow_hint(SAMPLE, "auto")


def test_binary_skips_non_binary() -> None:
    assert binary_arrow_hint("aa -> bb\ncc -> dd\n") is None


def test_none() -> None:
    assert augment_prompt_for_binary_arrow_hint(SAMPLE, "none") == SAMPLE
