"""
Hints for Wonderland **8-bit binary arrow** puzzles (bit manipulation families).

Prompts already mention shifts / XOR / etc.; we reinforce width-preservation from
the example pairs and echo which operator families the wording cites — structural only.
"""
from __future__ import annotations

import re
from typing import Literal

BinaryArrowStrategy = Literal["none", "auto"]

# Lines like "01111001 -> 11110010" (Wonderland CSV uses 8-bit tokens).
_PAIR_LINE = re.compile(r"^([01]{8})\s*->\s*([01]{8})\s*$")

_QUERY_OUT = re.compile(
    r"(?:determine|compute)\s+the\s+output\s+for\s*:\s*([01]{8})\b",
    re.IGNORECASE,
)
_QUERY_FALLBACK = re.compile(r"output\s+for\s*:\s*([01]{8})\b", re.IGNORECASE)


def _binary_arrow_pairs(prompt: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in prompt.splitlines():
        m = _PAIR_LINE.match(line.strip())
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs


def _query_bits(prompt: str) -> str | None:
    for pat in (_QUERY_OUT, _QUERY_FALLBACK):
        m = pat.search(prompt)
        if m:
            return m.group(1)
    return None


def _mentioned_ops_summary(prompt_lower: str) -> str | None:
    """Echo Wonderland CSV boilerplate about allowed instruction families (not the hidden rule)."""
    # Typical template: "... operations like bit shifts, rotations, XOR, AND, OR, NOT ..."
    if "bit shifts" in prompt_lower and "xor" in prompt_lower:
        bits = ["bit shifts", "rotations", "XOR", "AND", "OR", "NOT"]
        if "majority" in prompt_lower:
            bits.append("majority")
        if "choice" in prompt_lower:
            bits.append("choice")
        return ", ".join(bits)
    # Sparse fallback if wording changes
    fb: list[str] = []
    if "shift" in prompt_lower:
        fb.append("shifts")
    if "xor" in prompt_lower:
        fb.append("XOR")
    if not fb:
        return None
    return ", ".join(fb)


def _is_binary_family(prompt_lower: str) -> bool:
    return any(k in prompt_lower for k in ("binary", "8-bit", "8 bit", "bit manipulation"))


def binary_arrow_hint(prompt: str) -> str | None:
    pl = prompt.lower()
    if not _is_binary_family(pl):
        return None
    pairs = _binary_arrow_pairs(prompt)
    if len(pairs) < 2:
        return None
    widths = {len(a) for a, _ in pairs} | {len(b) for _, b in pairs}
    if len(widths) != 1:
        return None
    w = widths.pop()
    q = _query_bits(prompt)
    q_tail = f" Query bit-string length looks **{len(q)}** bits." if q else ""

    ops = _mentioned_ops_summary(pl)
    op_sentence = (
        f"The puzzle framing mentions these kinds of ops (hints only): {ops}. "
        if ops
        else ""
    )

    return (
        f"Binary arrow-rule hint: every listed example maps **{w}** bits → **{w}** bits "
        f"(preserve fixed width — reply with exactly **one** {w}-bit token, no spaces). "
        f"{op_sentence}"
        f"The implicit rule is the same deterministic mapping across all examples; extend it to the query input.{q_tail}"
    )


def augment_prompt_for_binary_arrow_hint(prompt: str, strategy: BinaryArrowStrategy) -> str:
    if strategy == "none":
        return prompt
    hint = binary_arrow_hint(prompt)
    if hint is None:
        return prompt
    return f"{prompt.rstrip()}\n\n{hint}"
