# Training Plan — 2026-05-12

This is the execution plan for tomorrow's training cycle, based on recent v8a/v9/v10 results and current access constraints.

## Current State

- Recent best runs are clustered around:
  - `v8a`: full `0.558`, first-token `0.665`
  - `v9`: full `0.558`, first-token `0.662`
  - `v10`: full `0.557`, first-token `0.658`
- Interpretation: prompt-hint tweaks are near a local plateau without stronger data improvements.

## Core Objective (Tomorrow)

Break plateau with higher-precision equation/transformation guidance and data targeting, while protecting non-equation performance.

## Experiment Plan (3 Runs)

1. **Run A — Evidence-Gated Equation Hint**
   - Keep current no-assistant-loss recipe (`assistant_loss_only=false`).
   - Equation hint only fires when:
     - exact phrase trigger matches (`"transformation rules is applied to equations"`), and
     - enough parseable equation evidence exists.
   - Keep hint compact (short, low-token noise).

2. **Run B — Rule-Scorer Equation Hint (Numeric Subset)**
   - For numeric equation rows, score and emit top candidate families:
     - normal ops: `+`, `-`, `*`, `concat` (with optional `±1`)
     - reverse/unreverse variants (with optional `±1`)
     - include multiply `+1` before unreverse candidate where supported.
   - Append only top 1-2 candidate families, not broad prose.

3. **Run C — Control**
   - Same as Run A but disable new equation add-on.
   - Purpose: isolate true gain from random variance.

## Evaluation Protocol

- Track both overall and bucketed performance:
  - overall: full and first-token
  - buckets: `equation_numeric`, `equation_symbolic`, `binary`, `phrase`, `roman`, `float`, `int`
- Promotion rule:
  - no overall full-accuracy drop > `0.5` points
  - clear gain in at least one equation bucket

## Synthetic Data Lift Plan (HF-Independent)

Because HuggingFace dataset access is currently unavailable, run a solver-centric synthetic pipeline using available local/remote training data.

1. Identify highest-loss transformation/equation failure buckets.
2. Generate rule-grounded synthetic examples for those buckets.
3. Verify each sample deterministically before inclusion.
4. Mix synthetic data conservatively (start ~20-40% of batch mix).
5. Continue with short equation-heavy continuation, then full-mixture retrain.

## HF Access Contingency

- Primary plan proceeds without HF datasets.
- In parallel, continue access escalation:
  - re-request access on dataset pages
  - contact dataset maintainer
  - open HF support ticket

## DGX Spark / Open Contribution Deliverables

To stay eligible for the Best Data/Synthetic Data contribution path:

- Produce a public Kaggle notebook with reproducible method.
- Include ablations and failure-bucket improvements.
- Prepare write-up narrative:
  - problem framing
  - solver/rule discovery method
  - synthetic data generation + filtering
  - measured gains and trade-offs
- Submit final `submission.zip` and complete Open Contribution submission requirements.

## Success Criteria for This Week

- At least one run beats or clearly matches `v8a` overall while improving equation buckets.
- First synthetic-data experiment demonstrates measurable bucket lift.
- Notebook/write-up scaffold created so evidence can be added incrementally.
