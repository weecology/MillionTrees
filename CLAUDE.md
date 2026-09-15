# CLAUDE.md

Project-wide instructions for Claude Code. `AGENTS.md` is the full contributor/agent
guide; this file surfaces the conventions that every session must follow so that
runs stay trackable and decisions stay fast. **Read these before submitting any job.**

## 1. SLURM job ledger (required on every `sbatch`)

The ledger at `/home/b.weinstein/logs/job_ledger.md` is how `checklog` (and a human
weeks later) explains *why* each run exists and *what comes next*. It is the single
source of truth for in-flight experiments.

**Every time you submit an `sbatch` job, immediately append one entry per job ID:**

```
## <JOBID> — <YYYY-MM-DD HH:MM> — <script name>
Why: <one line — the goal/hypothesis behind this run>
Next: <one line — which log to read, which dependent job to kick off, which table to update>
```

Rules:
- Append-only. Never edit or delete prior entries.
- Keep it to the two lines (`Why` / `Next`). For job arrays / chained submits, write one
  entry per job ID and use `Next` to record the dependency (e.g. "blocks 12345 polygon train").
- After submitting: write the ledger entry, then `ScheduleWakeup` ~60s, then
  `squeue -j <JOBID>` and check the `.out`/`.err` to confirm it's running (ST=R) with no errors.

A submit without a ledger entry is an incomplete submit — do not move on until it's written.

## 2. Informative Comet experiment names (required on every training run)

Comet auto-assigns random hashes (e.g. `3f87ee2d…`) when `experiment_name` is unset, which
makes the dashboard unreadable and slows decisions. **Every training run must set an explicit,
self-describing experiment name** following this scheme:

```
<geometry>-<split>-<init>-lr<lr>[-<purpose>]
```

- `geometry`: `boxes` | `points` | `polygons`
- `split`: `within-distribution` | `crossgeometry` | `out-of-distribution` (matches `--split-scheme`)
- `init`: `pretrained` | `scratch` (matches `--init-mode`)
- `lr`: learning rate, e.g. `lr0.005`
- `purpose` (optional): one short token for the hypothesis — `aug`, `smoke`, `overfit`,
  `baseline`, `sweep-a`, etc. Use this to distinguish otherwise-identical runs.

Examples: `polygons-within-distribution-pretrained-lr0.005-aug`, `points-within-distribution-pretrained-lr0.05`,
`boxes-crossgeometry-scratch-lr0.01-baseline`.

The experiment name should map back to its ledger `Why` so a name on the Comet board is
self-explanatory. Keep the existing `project_name` and `tags` as-is — the name is an
addition, not a replacement.

**Mechanism:** all three `training/{boxes,points,polygons}/train.py` accept a
`--comet-name` flag. If omitted, each builds a default from its own args following the
scheme above (e.g. polygons → `polygons-<split>-<init>-lr<lr>`), so you get an informative
name for free. Pass `--comet-name` explicitly only to add a `purpose` token
(`...-aug`, `...-smoke`, `...-sweep-a`) or otherwise override the default.

## 3. Regenerating the leaderboard includes the weak-supervision pretraining experiments

A leaderboard refresh for a new dataset version has **three** components, not two. All three
must be rerun against the new version and reported together:

1. **Fine-tuned models** — `training/` on the train splits (`training/slurm/submit_all_training.sh`).
2. **Pretrained existing models** — `existing_models/` on the test splits (`existing_models/slurm/submit_all_eval.sh`).
3. **Weak-supervision pretraining experiments** — the polygon Mask R-CNN ablation that compares an
   unsupervised box-pretrained backbone vs. the generic COCO/torchvision backbone, on both the
   within-distribution and out-of-distribution splits. This is **Table 6** in the manuscript
   ("Polygon task performance using unsupervised pretraining backbone versus an ImageNet backbone").

`slurm/submit_all.sh` currently launches only (1) and (2). The weak-supervision ablation (3) is a
separate two-stage submit — **do not skip it on a version bump**:

```bash
bash training/weak_supervision/submit_weak_supervision.sh
```

This chains: pretrain backbone on TreeBoxes+unsupervised (`pretrain_backbone.sbatch`, array over both
splits) → polygon train with that backbone (`train_polygons_box_pretrained.sbatch`, depends on
pretrain) → polygon train with COCO weights in parallel (`train_polygons_coco.sbatch`). Because the
loaders default to the latest `_versions_dict` key, these pick up the new version automatically (e.g.
v0.19) — confirm that key exists (see `[[project_bump_versions_dict_per_release]]`) before submitting.
Report the four-row Recall / Mask-aware Precision / AP40 comparison per split in
`notes/weak_supervision_pretraining_table.md` and the manuscript Table 6.

## 4. AP is AP40 (headline) + AP60 (strict complement) — there is no AP50

The headline AP in this project is **AP40**: average precision with predictions matched to ground
truth at **IoU 0.4**, the same IoU the recall and mask-aware precision behind F1 use, so AP and F1
agree on what counts as a match. Since 2026-08-22 `TreeBoxes`/`TreePolygons` also register **AP60**
(IoU 0.6) — same predictions, same ranking, stricter match — so a result file emits both
`Average AP40:` and `Average AP60:`. The parsers in `scripts/` read `Average AP40:`; add the AP60
pattern when a table needs the second column.

Consequences to remember:

- **AP40 and AP60 are reported together, and AP60 is never reported alone.** On its own it is
  unreadable (crown boundaries are ambiguous at IoU 0.6); its job is the AP40 → AP60 *drop*, which
  separates finding trees from delineating them. On the validation split boxes retain 6–53 % of
  AP40 at AP60 and spread widely; polygons all sit at 33–37 % and separate nobody.
- **Never reintroduce an AP50 column.** A result file that only has `Average AP50:` is a run made
  before 2026-08-04; it needs re-evaluation, not a table that mixes IoUs. `make_benchmark_table.py`
  deliberately prints `-` for those rather than silently falling back.
- **AP is scored with `max_detection_thresholds=[1, 10, 1000]` on every geometry.** Never leave
  this at torchmetrics' default: `None` resolves to `[1, 10, 100]`, which scores only the 100
  highest-scoring predictions per image. TreeBoxes ran that way until 2026-08-26 and box crops are
  dense (mean ~75-83 GT/image, p99 ~300), so ~20 % (OOD) to ~25 % (WD) of test ground truth was
  unreachable — an oracle predicting GT exactly scored AP40 0.80 / 0.75 instead of 1.0, with the
  loss concentrated in the dense sources (SelvaBox 28 %, OAM-TCD 28 %, Sun et al. 2022 62 %). It
  also made box and polygon AP incomparable, since TreePolygons always used 1000. **Box AP40/AP60
  from before 2026-08-26 needs re-scoring**; recall / mask-aware precision / F1 / counting MAE are
  unaffected (`DetectionAccuracy` has no per-image cap). On validation the shift is small
  (macro AP40 +0.000 to +0.005) because only 2.5 % of its GT was capped; the test splits move much
  more. `rescore_validation_ap.py` defaults to 1000 — pass `--max-detections 100` to reproduce old
  numbers.
- AP40 runs ~+0.04 to +0.11 above the old AP50 on the same predictions, so numbers are **not**
  comparable to pre-2026-08-04 tables. Older analysis docs keep their AP50 figures and say so.
- The historical AP50-vs-AP40 comparison that motivated this lives in
  `notes/ap50_vs_ap40_existing_models.md` (`scripts/make_ap_iou_table.py`, `outputs_ap_iou/` runs);
  the AP40-vs-AP60 validation comparison is `notes/leaderboard_validation.md` +
  `notes/validation_ap40_ap60.csv` (`scripts/rescore_validation_ap.py --ap-ious`).
- Any AP question at a *new* IoU is answered offline from a `--save-predictions` dump by
  `scripts/rescore_validation_ap.py`, not by a GPU re-run.

## 5. Analysis write-ups go in `notes/`, never in `docs/`

`docs/` is the published readthedocs site. Every page there must be listed in the toctree in
`docs/index.rst`, and CI builds with `-W`, so **any markdown file added to `docs/` without a
toctree entry fails the build**. Do not "fix" that by adding an internal report to the toctree —
benchmark users should not be reading diagnostic logs.

Put analysis reports, experiment diagnostics, generated table fragments and anything written to
navigate context in `notes/` (see `notes/README.md`). Scripts that emit a report default their
`--output` there. Figures still live in `docs/public/`; notes link to them as `../docs/public/...`.

## 6. Right-size `--mem` before every `sbatch`

The lab QOS `ewhite` caps **total concurrent memory at ~1508 GB** across all GPU jobs
(`sacctmgr show qos ewhite`). We have 10 GPUs but historically request 240–480 GB per job,
so memory — not GPUs — is what leaves jobs `PENDING (QOSGrpMemLimit)`. Every over-requested
GB blocks another job from starting. Most scripts are wildly over-provisioned (points at
480 G actually peak ~70–170 G; most eval jobs request 64–400 G and use <15 G).

**Part of every submit: check the memory request against history.**

```bash
scripts/slurm_mem_audit.py --check training/slurm/<script>.sbatch
```

- Exit 3 + `OVER-REQUEST` → lower the script's `#SBATCH --mem` to the printed number
  before submitting (commit the change).
- `needs a probe run` / `no history` → submit as-is; the probe records a real number for
  next time.
- `tight` → raise it.

**How the numbers are produced.** `scripts/slurm_mem_probe.sh` is `source`d at the top of
the canonical train/pretrain sbatch scripts. On exit it appends the true peak working set
(`anon + shmem` from the job cgroup, counted once even across `srun` tasks and dataloader
workers) to `/home/b.weinstein/logs/mem_ledger.csv`. `slurm_mem_audit.py` prefers that
ledger; for scripts that predate the probe it falls back to `sacct MaxRSS`, which the
cluster's `jobacct_gather/linux` plugin **over-reports 2–4×** (it sums per-process RSS), so
those rows are an upper bound only and flagged provisional.

- **New sbatch scripts that train/pretrain** must `source scripts/slurm_mem_probe.sh`
  right after the `cd` (see `training/slurm/train_boxes.sbatch`).
- Full picture / periodic review: `scripts/slurm_mem_audit.py` → `notes/slurm_memory_audit.md`.
- The cgroup `memory.peak` and `sacct` both include reclaimable page cache / slab and will
  trend toward `--mem` for any I/O-heavy job — never size against those.

## More

See `AGENTS.md` for repository structure, the leaderboard workflow
(`slurm/submit_all.sh` → `scripts/make_benchmark_table.py`), dataset packaging, and
geometry-specific details.
