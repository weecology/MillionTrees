# shellcheck shell=bash
# slurm_mem_probe.sh -- source this near the top of an sbatch script, after the
# `#SBATCH` block and `set -euo pipefail`, to record the job's real peak memory.
#
#   source scripts/slurm_mem_probe.sh
#
# It starts a background poller on the job's cgroup and, on script exit, appends
# one row to $MEM_LEDGER (default /home/b.weinstein/logs/mem_ledger.csv):
#
#   timestamp,jobid,jobname,req_mem_gb,peak_workingset_gb,cg_memory_peak_gb,elapsed_s,nodes,host
#
# peak_workingset_gb = max over the run of (anon + shmem) from the job-level
# cgroup memory.stat -- physical pages that must actually fit, counted once even
# across srun tasks and dataloader workers (unlike sacct MaxRSS, which sums
# per-process RSS and over-reports shared/forked memory 2-4x).
#
# cg_memory_peak_gb = cgroup memory.peak at exit; includes reclaimable page
# cache and slab, so it trends toward --mem for any I/O-heavy job. Recorded only
# for context -- do not size against it.
#
# Sizing rule used by scripts/slurm_mem_audit.py:
#   recommended --mem = ceil(1.3 * peak_workingset_gb + 8)   (rounded up to 8G)

_MEMPROBE_LEDGER="${MEM_LEDGER:-/home/b.weinstein/logs/mem_ledger.csv}"
_MEMPROBE_INTERVAL="${MEM_PROBE_INTERVAL:-20}"
_MEMPROBE_REL="$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)"
_MEMPROBE_JOBCG="/sys/fs/cgroup${_MEMPROBE_REL%%/step_*}"
_MEMPROBE_PEAKFILE="$(mktemp)"
_MEMPROBE_START="$(date +%s)"

if [ ! -r "${_MEMPROBE_JOBCG}/memory.stat" ]; then
  echo "slurm_mem_probe: no readable cgroup at ${_MEMPROBE_JOBCG} -- skipping" >&2
else
  ( peak=0
    while :; do
      cur=$(awk '/^anon /{a=$2} /^shmem /{s=$2} END{print a+s}' "${_MEMPROBE_JOBCG}/memory.stat" 2>/dev/null || echo 0)
      [ -n "$cur" ] && [ "$cur" -gt "$peak" ] 2>/dev/null && { peak=$cur; echo "$peak" > "${_MEMPROBE_PEAKFILE}"; }
      sleep "${_MEMPROBE_INTERVAL}"
    done ) &
  _MEMPROBE_PID=$!

  _memprobe_report() {
    local rc=$?
    kill "${_MEMPROBE_PID}" 2>/dev/null || true
    local peak_b elapsed cgpeak_b req_b
    peak_b=$(cat "${_MEMPROBE_PEAKFILE}" 2>/dev/null || echo 0)
    cgpeak_b=$(cat "${_MEMPROBE_JOBCG}/memory.peak" 2>/dev/null || echo 0)
    req_b=$(cat "${_MEMPROBE_JOBCG}/memory.max" 2>/dev/null || echo 0)
    elapsed=$(( $(date +%s) - _MEMPROBE_START ))
    rm -f "${_MEMPROBE_PEAKFILE}" 2>/dev/null || true
    [ -f "${_MEMPROBE_LEDGER}" ] || echo "timestamp,jobid,jobname,req_mem_gb,peak_workingset_gb,cg_memory_peak_gb,elapsed_s,nodes,host" > "${_MEMPROBE_LEDGER}"
    awk -v ts="$(date -Iseconds)" -v jid="${SLURM_JOB_ID:-local}" -v jn="${SLURM_JOB_NAME:-unknown}" \
        -v rb="$req_b" -v pb="$peak_b" -v cb="$cgpeak_b" -v el="$elapsed" \
        -v nn="${SLURM_JOB_NUM_NODES:-1}" -v hn="$(hostname -s)" \
        'BEGIN{printf "%s,%s,%s,%.1f,%.1f,%.1f,%d,%s,%s\n", ts,jid,jn, rb/1073741824, pb/1073741824, cb/1073741824, el, nn, hn}' \
        >> "${_MEMPROBE_LEDGER}"
    return $rc
  }
  trap _memprobe_report EXIT
fi
