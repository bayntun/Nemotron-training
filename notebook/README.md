# `notebook/`

Source for the **public Kaggle notebook** that doubles as our Best Data
Method Open Contribution Award submission (Path B for the DGX Spark).

## What will live here

- `submission.ipynb` — the public-facing notebook. Required sections:
  - Problem framing (rule-induction puzzles, why solver-driven data is the
    right inductive bias).
  - Per-category solver code with worked examples on **cracked unsolved-tail
    transformation puzzles** -- the headline visible novelty.
  - Solver -> teacher-CoT pipeline: rule prefixing, generation,
    grader-filtering. Show example traces.
  - Training recipe (PiSSA init, QLoRA, ZeRO-2, hyperparameters).
  - Ablation table: zero-shot baseline -> Phase 1 SFT -> Phase 2
    SFT+synthetic -> Phase 3 KD student, broken down by category.
  - Reproduction instructions: data versions, training configs, seeds,
    expected wall-clock per phase.

## Built from

The notebook will compose from this repo's modules:

- `eval.grader` (verbatim grader)
- `solvers.{transformation, bit_manipulation, cipher, ...}` (Phase 2 solvers)
- `teacher.deepseek_client` (CoT generation)
- `train.sft`, `train.kd` (training recipes)
- Output JSONLs from `eval.greedy_harness` (ablation rows)

## Status

Drafted in Phase 4 (days 29-39). This directory is intentionally empty until
then.
