"""
Structural hints for equation / opaque-operator few-shot rows.

Detects short `lhs = rhs` style examples and, when RHS string lengths are uniform,
nudges the model to match that output width. Skips encrypt/cipher prompts to avoid
colliding with decrypt rows that also mention 'rules'.
"""
from __future__ import annotations

import re
from typing import Literal

EquationStrategy = Literal["none", "auto"]

_LINE_EQ = re.compile(r"^([^=\n]{1,80})\s*=\s*([^\n]{1,80})\s*$")


def equation_shape_hint(prompt: str) -> str | None:
    pl = prompt.lower()
    if any(k in pl for k in ("encrypt", "decrypt", "cipher")):
        return None
    if "=" not in prompt:
        return None
    if not (
        "transformation rules" in pl
        or ("equation" in pl and "wonderland" in pl)
        or ("secret set" in pl and "equation" in pl)
    ):
        return None

    rhs_lens: list[int] = []
    for line in prompt.splitlines():
        m = _LINE_EQ.match(line.strip())
        if not m:
            continue
        rhs = m.group(2).strip()
        if not rhs:
            continue
        rhs_lens.append(len(rhs))

    if len(rhs_lens) < 2:
        return None

    u = sorted(set(rhs_lens))
    if len(u) == 1:
        return (
            f"Equation / operator puzzle hint: every listed output token string has length **{u[0]}**. "
            f"Shape-match your final answer to that length unless the last query clearly breaks the pattern."
        )

    return (
        "Equation / operator puzzle hint: listed output lengths vary across examples ("
        + ", ".join(str(x) for x in u[:6])
        + "). Match the **closest** example shape to the query expression before answering."
    )


def augment_prompt_for_equation_shape_hint(prompt: str, strategy: EquationStrategy) -> str:
    if strategy == "none":
        return prompt
    hint = equation_shape_hint(prompt)
    if hint is None:
        return prompt
    return f"{prompt.rstrip()}\n\n{hint}"
