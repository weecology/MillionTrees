#!/bin/bash
# Composite TreeFormer hyperparameter panels (density map + points vs image_size
# and integration radius) for selected sources. See density_panel.py and
# notes/treeformer_hyperparameters.md.
#SBATCH --job-name=mt_tf_density_panel
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees/existing_models/treeformer
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

# Packaged v0.18 CSVs still use the pre-rename split names (random/zeroshot);
# this private dir aliases them to within-/out-of-distribution and symlinks the
# real images, so no shared data is modified. See density_panel.py.
ROOT_DIR=${MT_ROOT:-/blue/ewhite/b.weinstein/src/MillionTrees/existing_models/treeformer/private_data}
CHECKPOINT=${CHECKPOINT:-weecology/deepforest-tree-point}
SPLIT=${SPLIT:-within-distribution}

uv run python density_panel.py \
  --checkpoint "$CHECKPOINT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --source "OSBS megaplot 2025" \
  --source "Ventura et al. 2022" \
  --image-sizes 448 896 1280 \
  --radii 1 2 \
  --score-thresh 0.1 \
  --output-dir outputs/density_panels

echo "=== Done. Panels in outputs/density_panels ==="
