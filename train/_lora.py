"""LoRA configuration aligned with docs/NEMOTRON_PLAN.md (Mamba mixer layers)."""

from __future__ import annotations

from peft import LoraConfig

# Public plan / tonghuikang-style layer subset for Nemotron-H hybrid.
MAMBA_LAYER_INDICES: tuple[int, ...] = (
    0,
    2,
    4,
    7,
    9,
    11,
    14,
    16,
    18,
    21,
    23,
    25,
    28,
    30,
    32,
    35,
    37,
    39,
    41,
    44,
    46,
    48,
    50,
)


def mixer_lora_target_patterns(layer_indices: tuple[int, ...] | None = None) -> list[str]:
    """Regex patterns matching Linear leaf names for mixer in/out projections."""
    idx = layer_indices if layer_indices is not None else MAMBA_LAYER_INDICES
    patterns: list[str] = []
    for i in idx:
        patterns.append(rf".*\.layers\.{i}\.mixer\.in_proj$")
        patterns.append(rf".*\.layers\.{i}\.mixer\.out_proj$")
    return patterns


def build_lora_config(
    *,
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    init_lora_weights: str = "pissa",
    layer_indices: tuple[int, ...] | None = None,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """PiSSA / LoRA on Nemotron-H Mamba mixer projections (subset of layers)."""
    modules = target_modules if target_modules is not None else mixer_lora_target_patterns(layer_indices)
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
        init_lora_weights=init_lora_weights,
    )
