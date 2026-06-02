#!/usr/bin/env bash
# Launch the full MillionTrees leaderboard for a new dataset version:
#   1. fine-tuned models (training/) on the train splits
#   2. pretrained existing_models/ on the test splits
#
# After all jobs finish, regenerate the leaderboard tables with:
#   uv run python scripts/make_benchmark_table.py --splits random zeroshot
#
# For a dependency-chained version that also auto-builds the table, use
# slurm/run_benchmark.sbatch instead.
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees

echo "=== Submitting fine-tuned training jobs ==="
bash training/slurm/submit_all_training.sh

echo "=== Submitting pretrained existing-model eval jobs ==="
bash existing_models/slurm/submit_all_eval.sh

echo ""
echo "Training + eval submitted. Panel figures are NOT included here."
echo "After training jobs finish, run either:"
echo "  sbatch slurm/visualize_finetuned.sbatch"
echo "or submit the full pipeline (training + eval + viz + table):"
echo "  sbatch slurm/run_benchmark.sbatch"
echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
