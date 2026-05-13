"""
Structural hints for equation / opaque-operator few-shot rows.

Strategies:
- ``auto``: strict phrase trigger + >=2 example lines; compact shape + short mode hint.
- ``auto_rulescored``: same as ``auto`` plus a one-line numeric rule fit (top-2 families)
  when examples are two-digit numeric operands.

Skips encrypt/cipher prompts. Only fires for the exact Wonderland equation phrase
(see ``_EQUATION_TRIGGER``) so other ``transformation`` prompts are untouched.
"""
from __future__ import annotations

import re
from typing import Literal

EquationStrategy = Literal["none", "auto", "auto_rulescored"]

_LINE_EQ = re.compile(r"^([^=\n]{1,80})\s*=\s*([^\n]{1,80})\s*$")
_LHS_BINOP = re.compile(r"^\s*(-?\d+)\s*([^0-9\s])\s*(-?\d+)\s*$")
_INT_RE = re.compile(r"^-?\d+$")
_EQUATION_TRIGGER = "transformation rules is applied to equations"

# One line — keeps token budget small (Run A).
_COMPACT_MODE = (
    "Eq puzzle: remapped ops (+−*concat ±1) | reverse digits then unreverse "
    "(mul may +1 before unreverse) | symbol→digit mapping ('=' delimiter fixed)."
)


def _rev2(n: int) -> int:
    s = f"{abs(n):02d}"
    r = int(s[::-1])
    return -r if n < 0 else r


def _unrev_int(n: int) -> int:
    sign = -1 if n < 0 else 1
    s = str(abs(n))
    return sign * int(s[::-1])


def _numeric_candidate_values(a: int, b: int) -> dict[str, int]:
    ar, br = _rev2(a), _rev2(b)
    return {
        "add": a + b,
        "sub": a - b,
        "mul": a * b,
        "concat": int(f"{a}{b}"),
        "rev_add": ar + br,
        "rev_sub": ar - br,
        "rev_mul": ar * br,
        "rev_concat": int(f"{ar}{br}"),
        "unrev_rev_add": _unrev_int(ar + br),
        "unrev_rev_sub": _unrev_int(ar - br),
        "unrev_rev_mul": _unrev_int(ar * br),
        "unrev_rev_mul_plus1": _unrev_int(ar * br + 1),
    }


def _score_numeric_rules(examples: list[tuple[int, str, int, int]]) -> list[tuple[str, int]]:
    scores: dict[str, int] = {}
    for a, _op, b, y in examples:
        vals = _numeric_candidate_values(a, b)
        for name, v in vals.items():
            if y == v:
                scores[name] = scores.get(name, 0) + 2
            elif y == v + 1 or y == v - 1:
                scores[name] = scores.get(name, 0) + 1
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked


def _shape_hint(rhs_lens: list[int]) -> str:
    u = sorted(set(rhs_lens))
    if len(u) == 1:
        return (
            f"Equation shape: listed outputs all length **{u[0]}**; match unless the query breaks the pattern."
        )
    return (
        "Equation shape: output lengths vary ("
        + ", ".join(str(x) for x in u[:6])
        + "); match the closest example shape to the query."
    )


def equation_shape_hint(prompt: str, strategy: EquationStrategy = "auto") -> str | None:
    if strategy == "none":
        return None
    pl = prompt.lower()
    if any(k in pl for k in ("encrypt", "decrypt", "cipher")):
        return None
    if "=" not in prompt:
        return None
    if _EQUATION_TRIGGER not in pl:
        return None

    rhs_lens: list[int] = []
    arith_examples: list[tuple[int, str, int, int]] = []
    n_example_lines = 0
    for line in prompt.splitlines():
        m = _LINE_EQ.match(line.strip())
        if not m:
            continue
        lhs = m.group(1).strip()
        rhs = m.group(2).strip()
        if not rhs:
            continue
        n_example_lines += 1
        rhs_lens.append(len(rhs))
        ml = _LHS_BINOP.match(lhs)
        if ml and _INT_RE.match(rhs):
            a = int(ml.group(1))
            op = ml.group(2)
            b = int(ml.group(3))
            y = int(rhs)
            arith_examples.append((a, op, b, y))

    # Run A/B evidence gate: at least two `lhs = rhs` example lines.
    if n_example_lines < 2:
        return None

    parts: list[str] = [_shape_hint(rhs_lens), _COMPACT_MODE]

    if strategy == "auto_rulescored" and len(arith_examples) >= 2:
        ranked = _score_numeric_rules(arith_examples)
        top = [name for name, sc in ranked[:2] if sc > 0]
        if top:
            parts.append("Numeric example-fit (try first): " + ", ".join(top) + ".")

    return "\n".join(parts)


def augment_prompt_for_equation_shape_hint(prompt: str, strategy: EquationStrategy) -> str:
    if strategy == "none":
        return prompt
    hint = equation_shape_hint(prompt, strategy=strategy)
    if hint is None:
        return prompt
    return f"{prompt.rstrip()}\n\n{hint}"
