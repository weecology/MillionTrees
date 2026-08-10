#!/bin/bash
#SBATCH --job-name=ofo_field
#SBATCH --output=logs/slurm/process_ofo_field_%j.out
#SBATCH --error=logs/slurm/process_ofo_field_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Regenerate the OFO field (Young et al. 2025) TreePoints dataset with
# tile-local point coordinates. The largest orthomosaic is ~65k x 105k px and
# tile_mission reads the whole raster into memory, so this needs bigmem.
# Uses the project .venv directly (DeepForest 2.1.0) so split_raster translates
# point annotations to per-tile coordinates; do NOT swap the env via uv sync.

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p logs/slurm

export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees/src:${PYTHONPATH:-}

FIELD=/orange/ewhite/DeepForest/OpenForestObservatory/field
OFO_ROOT=/orange/ewhite/DeepForest/OpenForestObservatory/missions_03

echo "=== process_ofo_field start: $(date) ==="

srun .venv/bin/python data_prep/process_ofo_field.py \
  --field_trees "$FIELD/OFO-field-trees-registered-to-drone.gpkg" \
  --output_dir "$FIELD" \
  --ofo_root "$OFO_ROOT"

echo "=== process_ofo_field done: $(date) ==="
