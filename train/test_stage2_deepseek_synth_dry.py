"""Dry-run smoke for stage2 script selection logic (no API)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_stage2_dry_run_selects_failures(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parent.parent
    ev = tmp_path / "eval.jsonl"
    rows = [
        {"id": "1", "prompt": "In Alice equation x", "gt": "1", "ok_full": True},
        {"id": "2", "prompt": "transformation rules is applied to equations\na=b", "gt": "2", "ok_full": False},
    ]
    ev.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "stage2_deepseek_verified_synth.py"),
            str(ev),
            "--dry-run",
            "--bucket",
            "equation_or_rule",
            "--limit",
            "10",
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "selected=1" in r.stdout
