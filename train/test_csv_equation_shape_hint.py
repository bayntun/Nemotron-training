from train.csv_equation_shape_hint import augment_prompt_for_equation_shape_hint, equation_shape_hint


def test_equation_uniform_rhs_len() -> None:
    p = """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
52-57 = 000
15-59 = 044
Now, determine the result for: 71-46
"""
    h = equation_shape_hint(p)
    assert h is not None
    assert "3" in h  # length 3 outputs
    assert augment_prompt_for_equation_shape_hint(p, "auto") != p


def test_equation_skips_encrypt() -> None:
    p = "In Alice's Wonderland, decrypt cipher text\na->b\n"
    assert equation_shape_hint(p) is None
