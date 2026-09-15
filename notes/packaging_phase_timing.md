# Packaging run phase timing: when is the data usable vs. when are the zips done?

**Question:** `data_prep/package_datasets.py` takes ~40h end to end, but training and
evaluation read the unpacked directories under `/orange/ewhite/web/public/MillionTrees/`,
not the archives. How early can leaderboard jobs be launched?

**Answer: the data phase is ~14–18h; the zip phase is another ~24–25h.** Launching
leaderboard jobs at the data/zip boundary recovers roughly **a full day** per release.

## Measured phases

Phase boundaries are recovered from directory mtimes under
`/orange/ewhite/web/public/MillionTrees/` cross-referenced against `sacct` start/end.
The last data write of a run is always `TreePolygons_supervised_<version>/masks`, the
final `copy_packaged_assets_from_full()` call before the zip loop.

| Run | Job | Start | Data phase | Zip phase | Total |
|---|---|---|---|---|---|
| v0.22 | 37002681 | 2026-07-13 15:18 | **14h10m** | 23h39m | 37h50m |
| v0.23 | 38693722 | 2026-08-04 12:51 | **17h45m** | 25h23m | 43h08m |
| v0.23 points re-zip | 38827912 | 2026-08-06 12:35 | 14h01m | 14h03m (4 archives only) | 28h04m |

So the zips are **62–63 %** of wall clock and produce nothing that local training or
evaluation consumes. The third row is the `MILLIONTREES_ZIP_GEOMETRIES=TreePoints`
targeted re-release, included to show the data phase is stable at ~14h even when only
one geometry is re-archived.

## Ordering inside the data phase

From `run()` in [package_datasets.py](../data_prep/package_datasets.py):

1. combine / dedup / geometry normalization / validation-tile filtering (in memory)
2. `copy_images` + `copy_masks` for the three full geometries — **~10–12h, the bulk**
3. `create_mini_datasets` / `create_small_datasets` (Mini*, Small*)
4. `process_splits_and_release` for Small, then full, then supervised — writes
   `within-distribution.csv`, `out-of-distribution.csv`, `crossgeometry.csv`, `RELEASE_*.txt`
5. `copy_packaged_assets_from_full` for the six `*_supervised_*` image and mask dirs —
   **~1.5–3h, and it runs AFTER the CSVs are written**
6. zip loop

Step 5 is the trap: **the split CSVs land ~2–3h before the data is actually complete.**
On v0.23 the full CSVs were written at t+14.7h but `TreePolygons_supervised/masks`
did not finish until t+17.8h. A readiness check that only looks for the CSVs will start
training against half-populated `*_supervised_*` image dirs — and the supervised dirs
are exactly what `include_unsupervised=False` (the loader default) reads.

## The reliable readiness signal

The zip loop begins only after every data write has completed, so **the appearance of
the first `*_<version>.zip`** is a filesystem-level proof that the data phase is done.
It is buffer-independent, unlike the `=== Zipping releases for:` stdout marker (which
also works, and is checked as a fallback).

[slurm/wait_for_release_data.sh](../slurm/wait_for_release_data.sh) implements this:
all 12 release dirs present with non-empty `images/` and `masks/` and all three split
CSVs, **plus** the zip-phase proof, **plus** a quiescence check that no data directory
has been modified recently. See also
[[project_inplace_repackage_corrupts_reads]] — reading images while they are still
being written surfaces as `PIL.UnidentifiedImageError`, not as a clean failure.

## v0.24 (job 40310045) projection

Started 2026-08-26 14:41:57.

| Estimate | Data ready |
|---|---|
| v0.22 pace (14h10m) | 2026-08-27 ~04:52 |
| v0.23 pace (17h45m) | 2026-08-27 ~08:27 |
| +20 % margin | 2026-08-27 ~11:42 |

Zips would not finish until ~2026-08-28 09:00. Polling should start around
2026-08-27 04:30 and the gate, not the clock, should trigger the submit.
