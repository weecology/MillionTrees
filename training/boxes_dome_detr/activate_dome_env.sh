#!/bin/bash
# Activate the Dome-DETR training environment
# Usage: source training/boxes_dome_detr/activate_dome_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOME_DETR_ROOT="/blue/ewhite/b.weinstein/src/Dome-DETR"

# Activate the isolated venv
source "$SCRIPT_DIR/.venv/bin/activate"

# Set PYTHONPATH so Dome-DETR source can be imported
export PYTHONPATH="$DOME_DETR_ROOT:$PYTHONPATH"

# Print confirmation
echo "✓ Dome-DETR environment activated"
echo "  Python: $(python --version)"
echo "  PYTHONPATH includes: $DOME_DETR_ROOT"
echo "  Ready to run: python training/boxes_dome_detr/train.py ..."
