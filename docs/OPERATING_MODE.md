# Operating Mode (Source of Truth)

Use this file to decide what to do next. If this file conflicts with older notes,
follow this file.

## Current mode

- **Mode:** Experimental hardening (pre-submission)
- **Meaning:** We are still validating training/eval reliability and holdout behavior.
- **Not yet in:** Final submission packaging phase.

## Phase gate definitions

- **Experimental phase:** iterate on data mix, training stability, holdout eval, and failure analysis.
- **Submission phase:** freeze recipe, train final adapter, run full holdout checks, package, and submit.

## Exit criteria to leave experimental phase

All must be true:

1. End-to-end remote run is stable (no SSH-drop loss, repeatable scripts).
2. Holdout eval completes and writes results reliably.
3. Symbolic subset report is generated and reviewed (correct + wrong rows).
4. One canonical training recipe is selected (no ambiguity about launcher/script).
5. Canonical file list is documented (which scripts are active vs archived).

When all five are true, switch this file to `Mode: Submission`.

## Canonical commands (current)

- Train launcher: `scripts/_launch_nemotron_sft_deepseek_remote.py`
- Holdout eval: `scripts/eval_nemotron_holdout_transformers.py`
- Symbolic report: `scripts/report_symbolic_holdout_eval.py`

## Immediate next actions

1. Finish one successful symbolic holdout eval run and save JSONL.
2. Generate symbolic success/failure report and record observations.
3. Keep non-canonical files archived under dated folders (done for 2026-05); apply the same rule to non-Python files.
4. Update this file to `Mode: Submission` only after the gate criteria pass.
