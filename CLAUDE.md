# CLAUDE.md

Project-wide instructions for Claude Code. `AGENTS.md` is the full contributor/agent
guide; this file surfaces the two conventions that every session must follow so that
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

## 4. AP is AP40 (IoU 0.4) — there is no AP50

Every reported AP in this project is **AP40**: average precision with predictions matched to
ground truth at **IoU 0.4**, the same IoU the recall and mask-aware precision behind F1 use, so
AP and F1 agree on what counts as a match. `TreeBoxes`/`TreePolygons` register a single `AP40`
metric (`src/milliontrees/datasets/`), and the parsers in `scripts/` read `Average AP40:`.

Consequences to remember:

- **Never reintroduce an AP50 column.** A result file that only has `Average AP50:` is a run made
  before this change; it needs re-evaluation, not a table that mixes IoUs. `make_benchmark_table.py`
  deliberately prints `-` for those rather than silently falling back.
- AP40 runs ~+0.04 to +0.11 above the old AP50 on the same predictions, so numbers are **not**
  comparable to pre-2026-08-04 tables. Older analysis docs keep their AP50 figures and say so.
- The historical AP50-vs-AP40 comparison that motivated this lives in
  `notes/ap50_vs_ap40_existing_models.md` (`scripts/make_ap_iou_table.py`, `outputs_ap_iou/` runs).

## 5. Analysis write-ups go in `notes/`, never in `docs/`

`docs/` is the published readthedocs site. Every page there must be listed in the toctree in
`docs/index.rst`, and CI builds with `-W`, so **any markdown file added to `docs/` without a
toctree entry fails the build**. Do not "fix" that by adding an internal report to the toctree —
benchmark users should not be reading diagnostic logs.

Put analysis reports, experiment diagnostics, generated table fragments and anything written to
navigate context in `notes/` (see `notes/README.md`). Scripts that emit a report default their
`--output` there. Figures still live in `docs/public/`; notes link to them as `../docs/public/...`.

## More

See `AGENTS.md` for repository structure, the leaderboard workflow
(`slurm/submit_all.sh` → `scripts/make_benchmark_table.py`), dataset packaging, and
geometry-specific details.
