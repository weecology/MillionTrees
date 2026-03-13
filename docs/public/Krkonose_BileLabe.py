#!/usr/bin/env python3
"""Runner for Krkonoše Bílé Labe processor. Run from repo root or with full path to this file.

  uv run python Krkonose_BileLabe.py /orange/ewhite/DeepForest/Zenodo_15591546 --visualize
  # or from Zenodo dir (extract_dir defaults to /orange/ewhite/DeepForest/Zenodo_15591546):
  uv run python /path/to/MillionTrees/Krkonose_BileLabe.py --visualize
"""
import sys
from pathlib import Path

# Run the real script from data_prep
_repo = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo / "data_prep"))

from Krkonose_BileLabe import main

if __name__ == "__main__":
    main()
