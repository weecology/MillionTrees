#!/bin/bash
# Generic sbatch template for running a script from a git worktree.
# Required env vars (pass via --export or set below):
#   WORKTREE  - absolute path to the git worktree
#   SCRIPT    - path to the script relative to WORKTREE
#   RUN_ARGS  - extra arguments forwarded to the script (optional)
#SBATCH --job-name=mt_pr_script
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --partition=hpg-default

set -euo pipefail

: "${WORKTREE:?WORKTREE env var is required}"
: "${SCRIPT:?SCRIPT env var is required}"

cd "$WORKTREE"

# Repo root and data_prep both on path so local modules (annotation_loop,
# label_studio_utils) and milliontrees package are importable.
export PYTHONPATH="$WORKTREE/data_prep:$WORKTREE:${PYTHONPATH:-}"

# Reuse the main venv — no re-install needed unless deps changed.
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/MillionTrees/.venv

uv run python "$SCRIPT" ${RUN_ARGS:-}
