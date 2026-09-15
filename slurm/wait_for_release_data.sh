#!/usr/bin/env bash
# Is a packaged MillionTrees release ready for training/eval, ignoring the zips?
#
# A packaging run spends ~14-18h writing data and another ~24h zipping (see
# notes/packaging_phase_timing.md). Training and eval read the unpacked directories,
# so they can start at the data/zip boundary and save ~a day.
#
# Usage:
#   bash slurm/wait_for_release_data.sh check [VERSION]   # exit 0 if ready, 1 if not
#   bash slurm/wait_for_release_data.sh wait  [VERSION]   # poll until ready
#
# Env: BASE_DIR (default /orange/ewhite/web/public/MillionTrees)
#      POLL_SECONDS (default 900), QUIESCE_SECONDS (default 300)
set -uo pipefail

MODE="${1:-check}"
VERSION="${2:-v0.24}"
BASE_DIR="${BASE_DIR:-/orange/ewhite/web/public/MillionTrees}"
POLL_SECONDS="${POLL_SECONDS:-900}"
QUIESCE_SECONDS="${QUIESCE_SECONDS:-300}"

# The six directories any leaderboard job can touch. The loader default is
# include_unsupervised=False, which reads the *_supervised_* dirs -- and those are
# populated LAST, after the split CSVs are already on disk.
DIRS=(
  "TreeBoxes_${VERSION}"        "TreePoints_${VERSION}"        "TreePolygons_${VERSION}"
  "TreeBoxes_supervised_${VERSION}" "TreePoints_supervised_${VERSION}" "TreePolygons_supervised_${VERSION}"
)
SPLIT_CSVS=(within-distribution.csv out-of-distribution.csv crossgeometry.csv)

fail() { echo "  NOT READY: $*"; return 1; }

check_ready() {
  local d p
  for d in "${DIRS[@]}"; do
    p="$BASE_DIR/$d"
    [ -d "$p" ] || { fail "missing dir $p"; return 1; }
    for csv in "${SPLIT_CSVS[@]}"; do
      [ -s "$p/$csv" ] || { fail "missing/empty $d/$csv"; return 1; }
    done
    [ -s "$p/RELEASE_${VERSION}.txt" ] || { fail "missing $d/RELEASE_${VERSION}.txt"; return 1; }
    for sub in images masks; do
      [ -d "$p/$sub" ] || { fail "missing $d/$sub"; return 1; }
      # -mindepth/-quit: cheap non-empty test on directories with ~1e6 entries.
      [ -n "$(find "$p/$sub" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ] \
        || { fail "$d/$sub is empty"; return 1; }
    done
  done

  # Zip-phase proof. zip_directory() runs strictly after every data write, so the
  # existence of any archive for this version means the data phase completed. This is
  # buffer-independent; the stdout marker is only a fallback.
  local zips
  zips=$(find "$BASE_DIR" -maxdepth 1 -name "*_${VERSION}.zip" -print -quit 2>/dev/null)
  if [ -z "$zips" ]; then
    local marker
    marker=$(grep -l "=== Zipping releases for" /home/b.weinstein/logs/format_MillionTrees_*.out 2>/dev/null \
             | xargs -r grep -l "$VERSION" 2>/dev/null | head -1)
    [ -n "$marker" ] || { fail "zip phase has not started (data phase still running)"; return 1; }
  fi

  # Quiescence: nothing under the release dirs modified in the last QUIESCE_SECONDS.
  # Guards against reading an image while it is still being written, which surfaces as
  # PIL.UnidentifiedImageError rather than a clean failure.
  local recent
  recent=$(find "${DIRS[@]/#/$BASE_DIR/}" -maxdepth 1 \
             -newermt "-${QUIESCE_SECONDS} seconds" -print -quit 2>/dev/null)
  [ -z "$recent" ] || { fail "still being written ($recent modified <${QUIESCE_SECONDS}s ago)"; return 1; }

  echo "  READY: $VERSION data complete under $BASE_DIR (zips may still be building)"
  return 0
}

case "$MODE" in
  check)
    echo "Checking $VERSION readiness in $BASE_DIR ..."
    check_ready
    ;;
  wait)
    echo "Polling for $VERSION every ${POLL_SECONDS}s ..."
    until check_ready; do
      echo "  $(date '+%F %T') not ready; sleeping ${POLL_SECONDS}s"
      sleep "$POLL_SECONDS"
    done
    ;;
  *)
    echo "usage: $0 {check|wait} [VERSION]" >&2; exit 2
    ;;
esac
