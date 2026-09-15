#!/bin/bash
# Poll for the v0.24 data phase to finish, then submit the whole leaderboard refresh.
#
# The packaging job (40310045) spends ~14-18h writing data and another ~24h zipping.
# Nothing downstream reads the zips, so this waits on the data/zip boundary and fires
# the submit there, recovering ~a day. See notes/packaging_phase_timing.md.
#
# Runs as a 1-CPU job so the trigger survives a closed laptop or ended session.
#   sbatch slurm/watch_and_submit_v024.sbatch
# Cancel with: scancel -n mt_watch_v024
#SBATCH --job-name=mt_watch_v024
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --account=ewhite
#SBATCH --qos=ewhite
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

VERSION="${VERSION:-v0.24}"
export POLL_SECONDS="${POLL_SECONDS:-900}"

echo "=== watcher started $(date '+%F %T') for $VERSION, polling every ${POLL_SECONDS}s ==="

bash slurm/wait_for_release_data.sh wait "$VERSION"

echo "=== $VERSION data ready at $(date '+%F %T'); submitting the leaderboard ==="

# The gate just passed, so --force skips a redundant re-check. The duplicate-job guard
# inside the launcher still runs and still aborts on a clash.
bash slurm/submit_leaderboard_v024.sh --force

echo "=== watcher done $(date '+%F %T') ==="
