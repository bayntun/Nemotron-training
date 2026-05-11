from train.csv_numeric_baseline import augment_prompt_for_numeric_baseline


def test_linear_meters_synthetic() -> None:
    p = (
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
        "For example: 1.0 m becomes 2.0 2.0 m becomes 4.0 3.0 m becomes 6.0 "
        "Now, convert the following measurement: 10.0 m"
    )
    out = augment_prompt_for_numeric_baseline(p, "linear_meters")
    assert "Numerical baseline" in out
    assert "20" in out or "20.0" in out


def test_gravity_quadratic_synthetic() -> None:
    # d = 3 * t^2
    p = (
        "In Alice's Wonderland, the gravitational constant has been secretly changed. "
        "For t = 1.0s, distance = 3.0 m For t = 2.0s, distance = 12.0 m "
        "Now, determine the falling distance for t = 3.0s given d = 0.5*g*t^2"
    )
    out = augment_prompt_for_numeric_baseline(p, "gravity_quadratic")
    assert "Numerical baseline" in out
    assert "27" in out or "27.0" in out


def test_none_unchanged() -> None:
    p = "In Alice's Wonderland, roman numeral puzzle: III -> V"
    assert augment_prompt_for_numeric_baseline(p, "none") == p
    assert augment_prompt_for_numeric_baseline(p, "auto") == p
