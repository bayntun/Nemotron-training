# `submit/`

Build the `submission.zip` Kaggle expects, with pre-zip validation to catch
the most common ways a submission silently fails.

## What lives here

- **`package.py`** — packs a PEFT LoRA adapter directory into
  `submission.zip` (two files at the zip root: `adapter_config.json` and
  `adapter_model.safetensors`). Validates `r <= 32`, `peft_type == "LORA"`,
  non-empty `target_modules`, and parseable safetensors before writing.

## Running

```powershell
python -m submit.package --adapter ./adapters/sft_baseline --out submission.zip
python -m submit.package --adapter ./adapters/kd_student   --out submission_kd.zip
```

To skip validation (not recommended):

```powershell
python -m submit.package --adapter ./adapters/x --out x.zip --no-validate
```

## What goes in the zip

The Kaggle scoring kernel loads the LoRA via PEFT/vLLM, so it only reads:

- `adapter_config.json`
- `adapter_model.safetensors`

Both at the zip's root (no nested directory). Anything else gets ignored.
