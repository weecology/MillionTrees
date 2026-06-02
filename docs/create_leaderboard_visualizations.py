"""Deprecated: use scripts/create_finetuned_visualizations.py instead."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_finetuned_visualizations.py"
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
