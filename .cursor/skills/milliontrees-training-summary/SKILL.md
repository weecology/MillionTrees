---
name: milliontrees-training-summary
description: Summarize latest MillionTrees training runs from results files and optional Lightning logs. Use when the user asks for training summary, latest runs, leaderboard-style metrics, or "how did training go."
---

# MillionTrees Training Summary

## When to use

Apply when the user wants a **summary of the latest MillionTrees training runs** (e.g. "summarize latest runs," "how did training go," "training summary," "latest metrics").

## What to do

1. **Results files**  
   Read summary metrics from the most recent `results_*.txt` under:
   - `training/points/outputs/{random,zeroshot,crossgeometry}/`
   - `training/boxes/outputs/{random,zeroshot,crossgeometry}/`
   - `training/polygons/outputs/{random,zeroshot,crossgeometry}/`  
   If multiple exist (e.g. `results_random.txt`, `results_zeroshot.txt`), use the latest by file mtime or include all.

2. **Extract** (from each file):
   - **Points**: "Average KeypointAccuracy", "Average CountingAccuracy" (or "Average counting_mae" / "Worst-group keypoint_acc" if present).
   - **Boxes**: "Average recall" (or "Avg Recall" from a summary block).
   - **Polygons**: "Average accuracy" or "Average recall", "Average mask_acc" if present.

3. **Optional — Lightning runs**  
   If the user wants training curves or final loss: under each `training/{geometry}/outputs/{split}/lightning_logs/`, take the latest `version_*` (by mtime or highest number). From that run’s `metrics.csv`, report final `val_loss` (last row with a val_loss) or best epoch.

4. **Optional — SLURM logs**  
   If the project has `logs/slurm/latest/` (symlink from `submit_all.sh`), list recent `.out`/`.err` files and mention any failures or last-completed jobs.

## Output format

Keep it short. Prefer a compact table, e.g.:

```text
| Geometry  | Split        | Main metric (e.g. KeypointAcc / Recall / mask_acc) | Optional val_loss |
| points    | random       | 0.212 KeypointAcc, 28.5 CountingMAE               | —                 |
| polygons  | zeroshot     | 0.183 recall                                      | —                 |
```

Then 1–2 lines on any SLURM failures or notable runs. If a path has no results yet, say "No results yet for {geometry}/{split}."

## Paths (project root)

- Results: `training/{boxes,points,polygons}/outputs/{random,zeroshot,crossgeometry}/results_*.txt`
- Lightning: `training/{geometry}/outputs/{split}/lightning_logs/version_*/metrics.csv`, `hparams.yaml`
- SLURM: `logs/slurm/latest/*.out`, `logs/slurm/latest/*.err`
