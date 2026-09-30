# MillionTrees SLURM job ledger

Project-local ledger read by `checklog`. Append-only; one entry per job ID:

```
## <JOBID> — <YYYY-MM-DD HH:MM> — <script name>
Why: <one line — goal/hypothesis>
Next: <one line — log to read / dependent job / table to update>
```

Jobs submitted before 2026-09-30 are recorded in the shared `/home/b.weinstein/logs/job_ledger.md`
(v0.25 leaderboard refresh: 42874753–42874768, plus resubmits 43652099–43652101).
Open items carried over from there:

## 43652099 — 2026-09-28 13:21 — training/slurm/pretrain_detectron2.sbatch (--array=1, OOD stage 1, --mem=64G)
Why: rerun of 42874764 (OOM at 40G); fresh v0.25 OOD stage-1 export needed for the box_pretrained_full arm.
Next: mt_pretrain_d2_43652099_1.{out,err}; unblocks 43652100. Result: PENDING

## 43652100 — 2026-09-28 13:21 — training/slurm/train_polygons_d2_ablation.sbatch (OOD, --array=1, afterok:43652099)
Why: replacement for cancelled 42874766; OOD box_pretrained_full Table 6 arm.
Next: training/polygons/outputs/detectron2_ablation/out-of-distribution/box_pretrained_full/; then val-F1 checkpoint sweep before Table 6. Result: PENDING

## 43652101 — 2026-09-28 13:21 — existing_models/slurm/eval_detectree2.sbatch (--array=1, OOD, --mem=400G)
Why: rerun of 42874759_1 (OOM at 200G); the OOD detectree2 row in docs/leaderboard.md is still the stale v0.24 number.
Next: existing_models/detectree2/outputs/out-of-distribution/results_polygons_out-of-distribution.txt, then regenerate leaderboard. Result: PENDING
