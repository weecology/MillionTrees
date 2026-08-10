#!/bin/bash
# Manuscript-final evaluation of the out-of-distribution models on the held-out
# Allen et al. 2025 TLS validation set, for all three geometries. Run ONCE when the
# final OOD checkpoints are in place. Each script auto-selects the best checkpoint
# by its monitored metric; override per-geometry with CKPT=... before sbatch.
#
# Remember (CLAUDE.md): append a job_ledger.md entry per returned job ID.
set -euo pipefail
cd "$(dirname "$0")"

sbatch eval_validation_boxes.sbatch
sbatch eval_validation_points.sbatch
sbatch eval_validation_polygons.sbatch
