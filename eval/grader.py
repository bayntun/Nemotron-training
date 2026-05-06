"""
Local copy of the NVIDIA Nemotron Reasoning Challenge scoring metric.

VERBATIM port of the `extract_final_answer` and `verify` functions from the
Kaggle "NVIDIA Nemotron Metric" notebook used by the competition's scoring
kernel. Do NOT modify the regexes, tolerances, or fallback ordering here --
even small deviations cause local eval to drift from leaderboard scores.

Source-of-truth: the metric notebook linked from the competition's Evaluation
page. Copy preserved by community RL training scripts (e.g. prometheus04's
train_grpo.py on HF) which explicitly mark this code as an EXACT copy.

Confirmed semantics:
- Numerical comparison: math.isclose(rel_tol=1e-2, abs_tol=1e-5)
- Both `stored_answer` and `predicted` are .strip()'d before comparison.
- String fallback is case-insensitive (.lower() == .lower()).
- Boxed-extraction prefers the LAST non-empty \\boxed{...} match.
- If no boxed match, falls back through "final answer" patterns, then last
  numeric value, then last non-empty line, then "NOT_FOUND".
"""

from __future__ import annotations

import math
import re


def extract_final_answer(text: str) -> str:
    r"""
    EXACT copy of competition's extract_final_answer function.
    Prioritizes \boxed{} content, falls back to heuristic patterns.
    """
    if text is None:
        return "NOT_FOUND"

    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            return found[-1].strip()

    found = re.findall(r'-?\d+(?:\.\d+)?', text)
    if found:
        return found[-1]

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """
    EXACT copy of competition's verify function.
    Numerical comparison with rel_tol=1e-2, otherwise string compare.
    """
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


def grade(generated_text: str, ground_truth: str) -> bool:
    """
    Convenience wrapper: extract from full generation, then verify.
    Use this in eval harnesses; use the underlying primitives in tests.
    """
    return verify(str(ground_truth), str(extract_final_answer(generated_text)))
