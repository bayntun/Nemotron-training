"""
Build submission.zip from a trained PEFT LoRA adapter directory.

The Kaggle scoring kernel expects a zip containing two files at the root:
    - adapter_config.json
    - adapter_model.safetensors

(matching the public solution's `zip -j` packaging).

Optional companion files are tolerated but ignored by vLLM:
    - adapter_model.bin (legacy format; prefer safetensors)
    - tokenizer*.json (the kernel uses the base model's tokenizer)

USAGE:
    python -m submit.package --adapter ./adapters/sft_baseline --out submission.zip
    python -m submit.package --adapter ./adapters/kd_student   --out submission_kd.zip --validate

The --validate flag runs sanity checks BEFORE writing the zip:
    - adapter_config.json exists and parses as JSON
    - adapter_config.json has r <= 32 (max_lora_rank enforcement)
    - adapter_config.json target_modules is non-empty
    - adapter_model.safetensors exists and parses as safetensors
    - peft_type is "LORA"
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REQUIRED_FILES = ["adapter_config.json", "adapter_model.safetensors"]
MAX_LORA_RANK = 32


def validate_adapter(adapter_dir: Path) -> dict:
    """Validate a PEFT LoRA adapter directory; return parsed adapter_config.json."""
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing adapter_config.json in {adapter_dir}. "
            "Make sure you saved with PeftModel.save_pretrained()."
        )

    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Missing adapter_model.safetensors in {adapter_dir}. "
            "Did the trainer save .bin instead? Convert to safetensors before packaging."
        )

    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)

    peft_type = config.get("peft_type")
    if peft_type != "LORA":
        raise ValueError(
            f"adapter_config.json has peft_type={peft_type!r}, expected 'LORA'. "
            "vLLM's LoRA loader only accepts standard LoRA adapters."
        )

    rank = config.get("r")
    if rank is None:
        raise ValueError("adapter_config.json missing 'r' (LoRA rank).")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError(f"adapter_config.json 'r' must be a positive int, got {rank!r}.")
    if rank > MAX_LORA_RANK:
        raise ValueError(
            f"adapter_config.json has r={rank}, exceeds the competition's "
            f"max_lora_rank={MAX_LORA_RANK}. Submission will be rejected by vLLM."
        )

    target_modules = config.get("target_modules")
    if not target_modules:
        raise ValueError("adapter_config.json has empty 'target_modules'.")

    # Optional but informative: spot-check the safetensors file is parseable.
    try:
        from safetensors import safe_open

        with safe_open(weights_path, framework="pt"):
            pass
    except ImportError:
        # safetensors isn't installed on the dev box; skip the sanity check.
        pass
    except Exception as e:
        raise ValueError(f"adapter_model.safetensors failed to parse: {e}") from e

    return config


def package(adapter_dir: Path, out_zip: Path, validate: bool = True) -> Path:
    """Build a submission.zip ready for upload to Kaggle."""
    if validate:
        config = validate_adapter(adapter_dir)
        print(
            f"Validated adapter: r={config['r']}, "
            f"alpha={config.get('lora_alpha')}, "
            f"target_modules={len(config['target_modules'])} layers, "
            f"task_type={config.get('task_type')}"
        )

    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in REQUIRED_FILES:
            src = adapter_dir / name
            if not src.is_file():
                raise FileNotFoundError(f"Missing required file: {src}")
            zf.write(src, arcname=name)

    size_mb = out_zip.stat().st_size / 1e6
    print(f"Wrote {out_zip} ({size_mb:.1f} MB)")
    return out_zip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--adapter", type=Path, required=True, help="Path to PEFT adapter directory")
    parser.add_argument("--out", type=Path, default=Path("submission.zip"), help="Output zip path")
    parser.add_argument("--no-validate", action="store_true", help="Skip pre-zip validation")
    args = parser.parse_args()

    try:
        package(args.adapter, args.out, validate=not args.no_validate)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
