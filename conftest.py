"""
Root pytest config: makes top-level packages importable when running `pytest`
from the repo root, so `from eval.grader import ...` resolves correctly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
