from train.csv_equation_shape_hint import augment_prompt_for_equation_shape_hint, equation_shape_hint


def test_equation_uniform_rhs_len() -> None:
    p = """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
52-57 = 000
15-59 = 044
Now, determine the result for: 71-46
"""
    h = equation_shape_hint(p, "auto")
    assert h is not None
    assert "Equation shape" in h
    assert "Eq puzzle" in h
    assert augment_prompt_for_equation_shape_hint(p, "auto") != p


def test_equation_skips_encrypt() -> None:
    p = "In Alice's Wonderland, decrypt cipher text\na->b\n"
    assert equation_shape_hint(p, "auto") is None


def test_equation_none_strategy() -> None:
    p = """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
9*4 = 36
8*5 = 40
Now, determine the result for: 7*6
"""
    assert equation_shape_hint(p, "none") is None


def test_evidence_gate_requires_two_lines() -> None:
    p = """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
9*4 = 36
Now, determine the result for: 7*6
"""
    assert equation_shape_hint(p, "auto") is None


def test_rulescored_adds_numeric_fit_line() -> None:
    p = """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
9*4 = 36
8*5 = 40
Now, determine the result for: 7*6
"""
    h = equation_shape_hint(p, "auto_rulescored")
    assert h is not None
    assert "Numeric example-fit" in h
    assert "mul" in h


def test_equation_does_not_trigger_without_exact_phrase() -> None:
    p = """In Alice's Wonderland, a secret set of transformation rules is applied to formulas. Below are a few examples:
9*4 = 36
8*5 = 40
Now, determine the result for: 7*6
"""
    assert equation_shape_hint(p, "auto") is None
    assert equation_shape_hint(p, "auto_rulescored") is None
