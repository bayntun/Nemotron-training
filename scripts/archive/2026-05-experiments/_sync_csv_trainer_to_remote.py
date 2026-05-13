#!/usr/bin/env python3
"""Stream a small tar of CSV-trainer files into remote docker (no git on server)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REMOTE = "tashpc-cursor"
CONTAINER = "jupyter-abayntun"
DEST = "/home/jovyan/work/Nemotron-training"
FILES = [
    "tmp_train_csv_remote.py",
    "train/csv_numeric_baseline.py",
    "train/csv_hint_gates.py",
    "train/csv_cipher_length_hint.py",
    "train/csv_encrypt_lexical_hint.py",
    "train/csv_equation_shape_hint.py",
    "train/csv_binary_arrow_hint.py",
    "train/csv_prompt_augment.py",
    "train/test_csv_numeric_baseline.py",
    "train/test_csv_cipher_length_hint.py",
    "train/test_csv_encrypt_lexical_hint.py",
    "train/test_csv_equation_shape_hint.py",
    "train/test_csv_binary_arrow_hint.py",
    "train/test_csv_hint_gates.py",
    "train/test_stage2_deepseek_synth_dry.py",
    "train/test_stage2_gemini_synth_dry.py",
    "scripts/run_v100_csv_train_4gpu.sh",
    "scripts/analyze_eval_failures.py",
    "scripts/analyze_hint_uptake.py",
    "scripts/check_symbol_mapping_consistency.py",
    "scripts/check_numeric_operator_rules.py",
    "scripts/stage2_deepseek_verified_synth.py",
    "scripts/stage2_gemini_verified_synth.py",
    "scripts/sample_transform_prompts.py",
    "scripts/_launch_csv_v6_encrypt_equation_remote.py",
    "scripts/_launch_csv_v7_binary_remote.py",
    "scripts/_launch_csv_v8_assistant_binary_remote.py",
    "scripts/_launch_csv_v8a_no_assistant_loss_remote.py",
    "scripts/_launch_csv_v8b_assistant_lowlr_remote.py",
    "scripts/_launch_csv_v9_eqhint_refresh_remote.py",
    "scripts/_launch_csv_v10_eqmode_strict_remote.py",
    "scripts/_launch_csv_v11a_eq_evidence_remote.py",
    "scripts/_launch_csv_v11b_eq_rulescore_remote.py",
    "scripts/_launch_csv_v11_eq_ab_sequential_remote.py",
    "scripts/_launch_csv_v12_rulescore_remote.py",
    "scripts/_launch_csv_v13_deepseek_merged_remote.py",
    "scripts/merge_synth_train_csv.py",
    "scripts/push_merged_train_csv_remote.py",
]


def main() -> int:
    recv = subprocess.Popen(
        [
            "ssh",
            REMOTE,
            "docker",
            "exec",
            "-i",
            CONTAINER,
            "tar",
            "xf",
            "-",
            "-C",
            DEST,
        ],
        stdin=subprocess.PIPE,
    )
    assert recv.stdin is not None
    send = subprocess.Popen(
        ["tar", "-cf", "-"] + FILES,
        cwd=REPO,
        stdout=recv.stdin,
    )
    send.wait()
    recv.stdin.close()
    recv.wait()
    if send.returncode != 0 or recv.returncode != 0:
        print(f"tar send={send.returncode} recv={recv.returncode}", file=sys.stderr)
        return 1
    print(f"Synced into {CONTAINER}:{DEST}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
