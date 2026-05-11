"""
Optional prompt augmentation for CSV few-shot Wonderland puzzles.

Fits *from the listed example pairs only*:
- unit-conversion style: linear map  y ≈ a*x + b  on (meters_in, meters_out)
- gravity-style tables:  d ≈ k * t^2  on (time_s, distance_m)

The real hidden rule may be non-linear; these hints are inductive bias for training
(and eval when the same flag is used), not ground truth.
"""
from __future__ import annotations

import re
from typing import Literal

Strategy = Literal["none", "auto", "linear_meters", "gravity_quadratic"]

_PAIR_M = re.compile(r"([\d.]+)\s*m\s+becomes\s+([\d.]+)", re.IGNORECASE)
_QUERY_M = re.compile(
    r"(?:Now,)?\s*convert\s+the\s+following\s+measurement\s*:\s*([\d.]+)\s*m",
    re.IGNORECASE,
)
_GRAVITY_ROW = re.compile(
    r"For\s+t\s*=\s*([\d.]+)\s*s\s*,\s*distance\s*=\s*([\d.]+)\s*m",
    re.IGNORECASE,
)
_GRAVITY_QUERY = re.compile(
    r"falling\s+distance\s+for\s+t\s*=\s*([\d.]+)\s*s",
    re.IGNORECASE,
)


def _linear_meters_hint(prompt: str) -> str | None:
    if "unit conversion" not in prompt.lower():
        return None
    pairs = _PAIR_M.findall(prompt)
    if len(pairs) < 2:
        return None
    xs = [float(a) for a, b in pairs]
    ys = [float(b) for a, b in pairs]
    mq = _QUERY_M.search(prompt)
    if not mq:
        return None
    x_q = float(mq.group(1))
    n = len(xs)
    s_x = sum(xs)
    s_y = sum(ys)
    s_xx = sum(x * x for x in xs)
    s_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * s_xx - s_x * s_x
    if abs(denom) < 1e-12:
        return None
    a = (n * s_xy - s_x * s_y) / denom
    b = (s_y - a * s_x) / n
    pred = a * x_q + b
    return (
        f"Numerical baseline (linear least-squares on the listed m→m′ pairs): {pred:.6g}. "
        f"If this disagrees with the implicit rule, follow the examples."
    )


def _gravity_quadratic_hint(prompt: str) -> str | None:
    if "gravitational" not in prompt.lower() and "falling distance" not in prompt.lower():
        return None
    rows = _GRAVITY_ROW.findall(prompt)
    if len(rows) < 2:
        return None
    ts = [float(t) for t, _d in rows]
    ds = [float(d) for _t, d in rows]
    mq = _GRAVITY_QUERY.search(prompt)
    if not mq:
        return None
    t_q = float(mq.group(1))
    # d ≈ k * t^2  (one parameter, closed form)
    num = sum((ti * ti) * di for ti, di in zip(ts, ds))
    den = sum((ti**4) for ti in ts)
    if abs(den) < 1e-18:
        return None
    k = num / den
    pred = k * t_q * t_q
    return (
        f"Numerical baseline (least-squares d=k·t² on the listed observations): {pred:.6g}. "
        f"If this disagrees with the implicit rule, follow the examples."
    )


def augment_prompt_for_numeric_baseline(prompt: str, strategy: Strategy) -> str:
    if strategy == "none":
        return prompt
    hint: str | None = None
    if strategy == "auto":
        hint = _linear_meters_hint(prompt)
        if hint is None:
            hint = _gravity_quadratic_hint(prompt)
    elif strategy == "linear_meters":
        hint = _linear_meters_hint(prompt)
    elif strategy == "gravity_quadratic":
        hint = _gravity_quadratic_hint(prompt)
    if hint is None:
        return prompt
    return f"{prompt.rstrip()}\n\n{hint}"
