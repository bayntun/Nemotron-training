"""
Download andy279 datasets from HuggingFace.

PREREQUISITES (manual, one-time):
    1. Create a HuggingFace account: https://huggingface.co/join
    2. Generate a read token: https://huggingface.co/settings/tokens
    3. Set HF_TOKEN in your .env (or environment).
    4. Visit each dataset page and CLICK "Agree and access repository":
         - https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge
         - https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge-raw-traces
       (Both repos are gated by a click-through ToS; without acceptance the
       download will 401 even with a valid token.)

USAGE:
    python -m data.download                    # downloads both datasets
    python -m data.download --sft-only         # only the cleaned SFT data
    python -m data.download --traces-only      # only the raw teacher traces

The cleaned SFT dataset (~408 MB) is sufficient for Phase 1 baseline reproduction.
The raw traces dataset (~1.02 GB) is needed for Phase 2 (re-verification, custom
filtering, and as the free teacher source for ~80% of training puzzles).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


SFT_REPO = "andy279/nemotron-reasoning-challenge"
TRACES_REPO = "andy279/nemotron-reasoning-challenge-raw-traces"

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache"


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN is not set. See data/download.py docstring for setup.",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def _download_dataset_repo(repo_id: str, dest: Path) -> Path:
    """Download an entire dataset repo (all files) via huggingface_hub.snapshot_download."""
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {dest} ...")
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        token=_hf_token(),
        max_workers=4,
    )
    print(f"  done: {local_path}")
    return Path(local_path)


def download_sft() -> Path:
    """Cleaned SFT dataset: 49,290 examples / 7,200 puzzles, ~408 MB."""
    return _download_dataset_repo(SFT_REPO, CACHE_DIR / "sft")


def download_traces() -> Path:
    """Raw teacher traces (DeepSeek + Nemotron Super 120B + solver-guided), ~1.02 GB."""
    return _download_dataset_repo(TRACES_REPO, CACHE_DIR / "traces")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sft-only", action="store_true", help="Only download cleaned SFT data")
    parser.add_argument("--traces-only", action="store_true", help="Only download raw traces")
    args = parser.parse_args()

    if args.sft_only and args.traces_only:
        parser.error("--sft-only and --traces-only are mutually exclusive")

    if not args.traces_only:
        download_sft()
    if not args.sft_only:
        download_traces()

    print("\nAll requested datasets downloaded into:", CACHE_DIR)


if __name__ == "__main__":
    main()
