#!/usr/bin/env python3
"""
Append synth rows (id,prompt,answer) to a base train CSV, deduping on ``id``.

By default, base rows win: synth rows whose ``id`` already appears in the base file are skipped.
Use ``--synth-overwrites`` to replace base rows when ``id`` collides.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="Existing train CSV (e.g. train_remote.csv)")
    ap.add_argument("--synth", type=Path, required=True, help="Synth CSV from stage2 (id,prompt,answer)")
    ap.add_argument("--out", type=Path, required=True, help="Output merged CSV path")
    ap.add_argument(
        "--synth-overwrites",
        action="store_true",
        help="If an id exists in both files, keep the synth row instead of the base row.",
    )
    args = ap.parse_args()

    with args.base.open(newline="", encoding="utf-8-sig") as f:
        base_reader = csv.DictReader(f)
        fieldnames = list(base_reader.fieldnames or [])
        if not fieldnames or "id" not in fieldnames or "prompt" not in fieldnames or "answer" not in fieldnames:
            print("ERROR: base CSV must have id, prompt, answer columns.", file=sys.stderr)
            return 2
        base_rows: list[dict[str, str]] = []
        seen: dict[str, dict[str, str]] = {}
        for row in base_reader:
            rid = str(row.get("id", "")).strip()
            base_rows.append(row)
            if rid:
                seen[rid] = row

    synth_rows: list[dict[str, str]] = []
    with args.synth.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "id" not in r.fieldnames:
            print("ERROR: synth CSV must have id (and prompt, answer).", file=sys.stderr)
            return 2
        for row in r:
            synth_rows.append(row)

    appended = 0
    skipped_dup = 0
    overwritten = 0
    out_rows: list[dict[str, str]] = []

    if args.synth_overwrites:
        synth_by_id = {str(row.get("id", "")).strip(): row for row in synth_rows if str(row.get("id", "")).strip()}
        used_synth: set[str] = set()
        for row in base_rows:
            rid = str(row.get("id", "")).strip()
            if rid and rid in synth_by_id:
                out_rows.append({k: synth_by_id[rid].get(k, "") for k in fieldnames})
                used_synth.add(rid)
                overwritten += 1
            else:
                out_rows.append({k: row.get(k, "") for k in fieldnames})
        for row in synth_rows:
            rid = str(row.get("id", "")).strip()
            if rid and rid not in seen and rid not in used_synth:
                out_rows.append({k: row.get(k, "") for k in fieldnames})
                appended += 1
    else:
        out_rows = [{k: row.get(k, "") for k in fieldnames} for row in base_rows]
        base_ids = {str(row.get("id", "")).strip() for row in base_rows if str(row.get("id", "")).strip()}
        for row in synth_rows:
            rid = str(row.get("id", "")).strip()
            if not rid:
                continue
            if rid in base_ids:
                skipped_dup += 1
                continue
            out_rows.append({k: row.get(k, "") for k in fieldnames})
            base_ids.add(rid)
            appended += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print(
        f"base_rows={len(base_rows)} synth_rows={len(synth_rows)} "
        f"out_rows={len(out_rows)} appended_new={appended} "
        f"skipped_duplicate_id={skipped_dup} overwritten={overwritten} "
        f"wrote={args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
