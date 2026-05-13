"""Heuristic for 'symbolic equation transform' holdout rows (not decrypt/binary/unit/gravity)."""


def is_symbolic_equation_holdout_prompt(prompt: str) -> bool:
    p = (prompt or "").lower()
    if "determine the result for" not in p:
        return False
    if "applied to equations" not in p:
        return False
    for bad in ("decrypt", "8-bit binary", "unit conversion", "gravitational constant"):
        if bad in p:
            return False
    return True
