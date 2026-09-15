#!/usr/bin/env python3
"""Audit SLURM memory requests against what jobs actually use.

Two data sources, in order of trust:

1. logs/mem_ledger.csv  -- written by scripts/slurm_mem_probe.sh. `peak_workingset_gb`
   is anon+shmem from the job cgroup: real physical pages, counted once even across
   srun tasks / dataloader workers. This is the number to size against.

2. sacct MaxRSS         -- fallback for jobs that ran before the probe existed. The
   cluster uses jobacct_gather/linux, which SUMS per-process RSS, so multi-worker
   jobs over-report 2-4x. Treated as an UPPER BOUND only: if even this is well
   below the request, the request is safe to cut; if it exceeds the request on a
   job that completed, the true figure is unknown (flagged `?`).

Usage:
  scripts/slurm_mem_audit.py                      # full report -> notes/slurm_memory_audit.md
  scripts/slurm_mem_audit.py --since 2026-07-01
  scripts/slurm_mem_audit.py --check training/slurm/train_points_896.sbatch
      -> prints recommended --mem for that script's job-name; exit 3 if the script
         over-requests by more than --slack (default 1.5x) and we have >=2 samples.

The sizing rule (headroom for page cache the kernel wants + spikes between polls):
  recommended = roundup8( 1.3 * peak_workingset + 8 )
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = Path("/home/b.weinstein/logs/mem_ledger.csv")
REPORT = REPO / "notes" / "slurm_memory_audit.md"


def roundup8(x: float) -> int:
    return int(math.ceil(x / 8.0) * 8)


def recommend(peak_ws_gb: float) -> int:
    return max(16, roundup8(1.3 * peak_ws_gb + 8))


def parse_mem_to_gb(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    m = re.match(r"^([\d.]+)\s*([KMGT]?)B?$", s, re.I)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2).upper()
    return val / 1048576 if unit == "K" else val / 1024 if unit == "M" else val * 1024 if unit == "T" else val


# ----------------------------------------------------------------------------- ledger

def load_ledger() -> dict[str, list[dict]]:
    by_name: dict[str, list[dict]] = defaultdict(list)
    if not LEDGER.exists():
        return by_name
    with LEDGER.open() as fh:
        for row in csv.DictReader(fh):
            try:
                row["peak_workingset_gb"] = float(row["peak_workingset_gb"])
                row["req_mem_gb"] = float(row["req_mem_gb"])
            except (KeyError, ValueError):
                continue
            if row["peak_workingset_gb"] > 0:
                by_name[row["jobname"]].append(row)
    return by_name


# ----------------------------------------------------------------------------- sacct

def load_sacct(since: str) -> dict[str, list[dict]]:
    try:
        out = subprocess.run(
            ["sacct", "--user", "b.weinstein", "--starttime", since, "--endtime", "now",
             "--format", "JobID,JobName%60,State,ReqMem,MaxRSS", "-P"],
            capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"sacct unavailable: {exc}", file=sys.stderr)
        return {}
    jobs: dict[str, dict] = {}
    for line in out.splitlines()[1:]:
        f = line.split("|")
        if len(f) < 5:
            continue
        jid, name, state, reqmem, maxrss = f[:5]
        base = jid.split(".")[0]
        step = jid.split(".")[1] if "." in jid else None
        j = jobs.setdefault(base, {"name": name, "state": state, "req": None, "peak": 0.0})
        if step is None:
            j["req"] = parse_mem_to_gb(reqmem)
            j["state"] = state
            j["name"] = name or j["name"]
        elif step.isdigit() or step == "batch":
            g = parse_mem_to_gb(maxrss)
            if g:
                j["peak"] = max(j["peak"], g)
    by_name: dict[str, list[dict]] = defaultdict(list)
    for j in jobs.values():
        if j["name"].startswith("mt_") and j["peak"] > 0:
            by_name[j["name"]].append(j)
    return by_name


# ----------------------------------------------------------------------------- sbatch scan

def scan_sbatch() -> dict[str, tuple[Path, float]]:
    """job-name -> (sbatch path, current --mem in GB)."""
    out: dict[str, tuple[Path, float]] = {}
    for d in ("training/slurm", "existing_models/slurm", "slurm"):
        for p in sorted((REPO / d).glob("*.sbatch")):
            txt = p.read_text()
            nm = re.search(r"#SBATCH\s+--job-name[= ]+(\S+)", txt)
            mm = re.search(r"#SBATCH\s+--mem[= ]+(\S+)", txt)
            if nm and mm:
                out.setdefault(nm.group(1), (p, parse_mem_to_gb(mm.group(1)) or 0.0))
    return out


# ----------------------------------------------------------------------------- report

def build_rows(since: str):
    ledger, sacct = load_ledger(), load_sacct(since)
    sbatch = scan_sbatch()
    names = sorted(set(ledger) | set(sacct))
    rows = []
    for name in names:
        lg = ledger.get(name, [])
        sc = sacct.get(name, [])
        cur_mem = sbatch.get(name, (None, None))[1]
        path = sbatch.get(name, (None, None))[0]
        if lg:
            peaks = sorted(r["peak_workingset_gb"] for r in lg)
            src, n = "cgroup", len(peaks)
            p95 = peaks[min(len(peaks) - 1, int(0.95 * len(peaks)))]
            rec = recommend(p95)
            note = ""
        else:
            peaks = sorted(j["peak"] for j in sc)
            src, n = "sacct~", len(peaks)
            p95 = peaks[min(len(peaks) - 1, int(0.95 * len(peaks)))]
            req = max((j["req"] for j in sc if j["req"]), default=None)
            if req and p95 > req:
                # sacct's inflated figure already exceeds the (working) request:
                # real usage is somewhere below the request, unknowable from here.
                rec, note = None, "sacct over-reports past request; ran fine — needs a probe run"
            else:
                # sacct is already an over-count, so don't inflate further; just
                # round the observed ceiling up a little. Never used to raise.
                rec, note = max(16, roundup8(1.15 * peaks[-1] + 8)), "upper bound (provisional; source is inflated)"
        rows.append(dict(name=name, path=path, cur_mem=cur_mem, src=src, n=n,
                         peak=p95, peak_max=peaks[-1], rec=rec, note=note))
    return rows


def fmt_report(rows, since: str) -> str:
    L = [
        "# SLURM memory audit",
        "",
        f"Generated by `scripts/slurm_mem_audit.py` (sacct since {since}).",
        "",
        "- **peak** = p95 of observed peak working set per job-name. Source `cgroup` = "
        "anon+shmem from `scripts/slurm_mem_probe.sh` (trustworthy). Source `sacct~` = "
        "MaxRSS, which sums per-process RSS and **over-reports 2-4x** for multi-worker "
        "jobs -- an upper bound only.",
        "- **rec** = `roundup8(1.3 * peak + 8)` for `cgroup`, `roundup8(1.15 * peak_max "
        "+ 8)` for the already-inflated `sacct~`. `?` = sacct over-reports past the "
        "request on a job that completed, so the real figure needs a probe run.",
        "- Lab QOS `ewhite` caps concurrent memory at **1508 GB** across all GPU jobs "
        "(`sacctmgr show qos ewhite`), so every over-requested GB blocks another job "
        "(`squeue` reason `QOSGrpMemLimit`).",
        "",
        "| job-name | sbatch --mem | peak (p95) | peak (max) | src | n | recommended | note |",
        "|---|--:|--:|--:|---|--:|--:|---|",
    ]
    for r in sorted(rows, key=lambda r: -(r["cur_mem"] or 0)):
        cur = f"{r['cur_mem']:.0f}G" if r["cur_mem"] else "-"
        rec = f"**{r['rec']}G**" if r["rec"] else "?"
        flag = ""
        if r["rec"] and r["cur_mem"] and r["cur_mem"] > 1.5 * r["rec"]:
            flag = " over-request"
        L.append(f"| {r['name']} | {cur} | {r['peak']:.0f}G | {r['peak_max']:.0f}G | "
                 f"{r['src']} | {r['n']} | {rec} | {r['note']}{flag} |")
    L += ["",
          "## Biggest wins",
          ""]
    wins = [r for r in rows if r["rec"] and r["cur_mem"] and r["cur_mem"] > 1.5 * r["rec"]]
    for r in sorted(wins, key=lambda r: -(r["cur_mem"] - r["rec"])):
        L.append(f"- `{r['name']}`: {r['cur_mem']:.0f}G -> {r['rec']}G "
                 f"(save {r['cur_mem'] - r['rec']:.0f}G/job){' — ' + str(r['path'].relative_to(REPO)) if r['path'] else ''}")
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------- check mode

def check(target: str, since: str, slack: float) -> int:
    rows = {r["name"]: r for r in build_rows(since)}
    p = Path(target)
    name = None
    if p.exists():
        m = re.search(r"#SBATCH\s+--job-name[= ]+(\S+)", p.read_text())
        name = m.group(1) if m else None
    name = name or target
    r = rows.get(name)
    if not r:
        print(f"[mem-audit] {name}: no history — leave --mem as is, first run will record it")
        return 0
    if not r["rec"]:
        print(f"[mem-audit] {name}: {r['n']} run(s), sacct over-reports past request — "
              f"run once with slurm_mem_probe.sh sourced to get a real number")
        return 0
    cur = r["cur_mem"]
    verdict = "OK"
    rc = 0
    if cur and cur > slack * r["rec"] and r["n"] >= 2:
        verdict = f"OVER-REQUEST — cut to {r['rec']}G"
        rc = 3
    elif cur and cur < r["rec"] and r["src"] == "cgroup":
        verdict = f"tight — consider raising to {r['rec']}G"
    print(f"[mem-audit] {name}: --mem={cur:.0f}G, observed peak p95={r['peak']:.0f}G "
          f"(max {r['peak_max']:.0f}G, {r['src']}, n={r['n']}), recommended {r['rec']}G -> {verdict}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-07-01")
    ap.add_argument("--check", metavar="SBATCH_OR_JOBNAME")
    ap.add_argument("--slack", type=float, default=1.5,
                    help="flag over-request when --mem exceeds slack * recommended")
    ap.add_argument("--output", type=Path, default=REPORT)
    args = ap.parse_args()

    if args.check:
        return check(args.check, args.since, args.slack)

    rows = build_rows(args.since)
    args.output.write_text(fmt_report(rows, args.since))
    print(f"wrote {args.output}  ({len(rows)} job-names)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
