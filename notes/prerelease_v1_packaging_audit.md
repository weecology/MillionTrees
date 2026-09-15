# Pre-v1 packaging audit: splits, logic errors, benchmark-quality concerns

**Update 2026-08-26:** findings #1, #2, and #4 are fixed (see commits touching
`data_prep/package_datasets.py`, `src/milliontrees/datasets/Tree{Boxes,Points,Polygons}.py`,
`tests/test_split_integrity.py`, `tests/test_release.py`, `tests/conftest.py`). Finding #3's
code fix was already in place (v0.24 commit `63d5a1e`); the v0.24 packaging job was restarted
after these fixes landed so the shipped data incorporates all of them in one pass. Findings
#5-#8 are unchanged and still open.

Full pass over `data_prep/package_datasets.py`, the loaders, and packaging tests, done in
preparation for a first stable release. Findings ranked critical -> low. Evidence for each
was checked directly against code and, where possible, against the packaged v0.23 CSVs at
`/orange/ewhite/web/public/MillionTrees/`.

## Summary table

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | **Critical** | Within-distribution auto-split cuts at annotation-row boundaries, not image boundaries -> same image split across train and test | **New, unfixed** |
| 2 | **Critical** | No regression test asserts train/test filename disjointness for any split scheme | **New, unfixed** |
| 3 | **High** | Out-of-distribution split leaks (SelvaBox, Cloutier, etc.) | Code fixed in v0.24, **not yet repackaged** |
| 4 | **High** | `mini=True` silently discarded when `include_unsupervised=False` (the default) in all three loaders | Known (memory), **unfixed** |
| 5 | **Medium** | Point OOD test macro-average gives the NEON sampling program 3 of 5 source votes at overlapping sites | Known limitation, documented, not blocking |
| 6 | **Medium** | Several packaging functions read a module-level `version` global instead of a parameter | **New, unfixed**, low probability of triggering |
| 7 | **Medium** | Sources with an upstream `existing_split` pin are trusted without verifying spatial disjointness of train/test tiles | Spot-checked (SelvaBox) and looks fine; not audited for the rest |
| 8 | **Low** | `docs/dataset_structure.md` describes within-distribution splitting at image granularity when the code (bug #1) does not guarantee that | Documentation gap, fix alongside #1 |

---

## 1. CRITICAL — Within-distribution split leaks images between train and test

`within_distribution_split.apply_split()` (`data_prep/package_datasets.py:1091-1108`) assigns
the auto-split fraction like this:

```python
for source in remaining["source"].dropna().unique():
    idx = remaining[remaining["source"] == source].index
    n = len(idx)
    n_test = max(1, int(round(n * 0.1))) if n > 1 else 0
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    ...
```

`idx` is the row index of **annotations**, not images. An image with many trees contributes
many rows; the cut at position `n_test` falls wherever it falls in row order, with no regard
for where one image's rows end and the next begins. Whichever image's annotations straddle
that boundary gets some rows in test and the rest in train — the same picture appears on both
sides of the eval.

**Confirmed on the packaged v0.23 CSVs** (same code path ships unchanged for v0.24, not yet
repackaged):

```
TreeBoxes:    13 images split across train/test (13 sources affected)
TreePoints:    7 images split across train/test (7 sources affected)
TreePolygons: 22 images split across train/test (18 sources affected)
```

Example — `108_Zamboni_et_al_2021.png` in `TreeBoxes_v0.23/within-distribution.csv`: rows
327-334 (8 boxes) are `test`, rows 335-349 (15 boxes) of the **same image** are `train`. The
cut point (row 335) is exactly `n_test` for the Zamboni source (335 of 3,353 rows = 10%).

This is not the same bug as the OOD leak in `notes/ood_split_test_sources_and_leaks.md` —
that one was tracked there (§8b) as "same filename in both train and test... likely rows of
one image disagreeing on `existing_split`, or a filename collision," with the true cause left
open. It is this row-slicing logic, and it explains the exact counts already recorded there
(13/7/22).

**Scope:** only sources that reach the auto-split branch (i.e. do *not* already carry an
upstream `existing_split` train/test pin) are exposed — that's most sources. It hits
`within-distribution.csv` in the **full** and **Small** releases (both call
`process_splits_and_release` -> `within_distribution_split`). **Mini is not affected**: it's
built earlier in `run()` from the image-level split already assigned by `split_dataset()`
(line 1561), before this function runs.

Per-source impact is small (about one image per affected source, since annotation density
means the row cut rarely coincides with an image boundary), but it is a genuine train/test
leak in the flagship split that every fine-tuned leaderboard entry uses, and one image
straddling both sides is one image too many for a benchmark whose job is to measure
generalization.

### Fix

Split at the image level, then propagate to rows, mirroring the existing correct pattern in
`split_dataset()` (`package_datasets.py:332-338`):

```python
for source in remaining["source"].dropna().unique():
    src_mask = remaining["source"] == source
    filenames = remaining.loc[src_mask, "filename"].unique()
    n = len(filenames)
    n_test = max(1, int(round(n * 0.1))) if n > 1 else 0
    test_filenames = filenames[:n_test]
    is_test = remaining["filename"].isin(test_filenames) & src_mask
    df.loc[remaining.index[is_test.reindex(remaining.index, fill_value=False)], "split"] = "test"
    ...
```

(Exact implementation detail aside, the invariant is: decide split membership on unique
`filename`, not on row position, then assign every row for a chosen filename to the same
split.) Also consider seeding/shuffling `filenames` before slicing — right now the 10% test
images are always "whichever images happen to be first in row order for that source," which
is at least deterministic but not a random sample; a fixed-seed shuffle would be a better
within-distribution test sample without changing the reproducibility property.

---

## 2. CRITICAL — No regression test for split disjointness

`tests/test_release.py` and `tests/test_TreeBoxes.py` / `test_TreePoints.py` /
`test_TreePolygons.py` check dataset shapes, loader batching, and metric correctness, but
nothing checks that `train` and `test` (or `validation`) don't share a filename. That's why
bug #1 shipped across at least the v0.23 release undetected, and why the OOD leak (finding
#3) needed a manual audit to surface rather than failing CI.

### Fix

Add a packaging-level test (can run against the packaged CSVs directly, no GPU/loader
needed) asserting, per geometry and per split scheme:

```python
def test_no_filename_leak_across_splits(csv_path):
    df = pd.read_csv(csv_path, usecols=["filename", "split"])
    counts = df.groupby("filename")["split"].nunique()
    leaked = counts[counts > 1]
    assert leaked.empty, f"{len(leaked)} filenames appear in >1 split: {leaked.index[:5].tolist()}"
```

Run it over `within-distribution.csv`, `out-of-distribution.csv`, and `crossgeometry.csv` for
all three geometries, both the full and Small releases. This is cheap, catches both bug #1
and any regression of bug #3, and should gate every future repackage.

---

## 3. HIGH — Out-of-distribution split leak (already coded, not yet shipped)

Full detail in `notes/ood_split_test_sources_and_leaks.md`. Verified against the current
code (`_assign_ood_split_by_source`, `OUT_OF_DISTRIBUTION_TEST_SOURCES_BOXES`,
`OUT_OF_DISTRIBUTION_TRAIN_ONLY_SOURCES` at `package_datasets.py:29-77, 921-934`): **the fix
described in that note's §11 is implemented correctly** in the v0.24 commit
(`63d5a1e`). SelvaBox is now `OUT_OF_DISTRIBUTION_TRAIN_ONLY_SOURCES`, OOD membership is
decided by source alone via `_assign_ood_split_by_source` (no more `existing_split` gate
exemption), and `cross_geometry_split` now drops box/point rows from declared polygon
hold-out sources before assigning them to train.

**But `/orange/ewhite/web/public/MillionTrees/` only has v0.23 packaged.** The fix is
sitting in code, unreleased. Per `CLAUDE.md` §3, a version bump needs: repackage (this is the
long pole — `notes/packaging_phase_timing.md` says ~25h for a full run), confirm the new
`_versions_dict` key exists in all three loaders (`[[project_bump_versions_dict_per_release]]`
memory), and rerun all three leaderboard components. Until that happens, anyone pulling
"latest" still gets the SelvaBox/Cloutier/OFO leaks.

**Recommendation:** bundle the repackage with the fix for finding #1 (both live in the same
script, same output files) rather than repackaging twice.

---

## 4. HIGH — `mini=True` silently ignored (all three geometries)

Confirmed still present in current code, `src/milliontrees/datasets/TreeBoxes.py:172-197`
(identical pattern in `TreePoints.py` / `TreePolygons.py`):

```python
if mini:
    self._versions_dict = self._get_mini_versions_dict()
elif small:
    self._versions_dict = self._get_small_versions_dict()

if not include_unsupervised:          # True by default
    ...
    if small:
        self._dataset_name = 'SmallTreeBoxes'
    else:
        self._dataset_name = 'TreeBoxes_supervised'   # <- mini=True falls here too
...
self._dataset_name = 'TreeBoxes'      # unconditional "restore" right after
```

When `mini=True, small=False, include_unsupervised=False` (the common "give me a fast smoke
test" call), `_dataset_name` never becomes anything Mini-specific — it's forced to
`'TreeBoxes_supervised'` and then unconditionally back to `'TreeBoxes'`. A `--mini` smoke run
downloads/reads the full release. This was confirmed empirically for polygons on 2026-08-25
(job 40200530: registered 9,315 train / 1,742 test tiles — the full v0.23 release — and timed
out at 1h because the "smoke" evaluated the full test set). Already in memory
(`project_mini_ignored_when_supervised`); flagged here because it's user-facing and
documented (`docs/dataset_structure.md`, `docs/getting_started.md` both advertise `mini=True`
as the fast path) — a v1 release should not ship a documented flag that silently no-ops.

### Fix

`_dataset_name` should carry the Mini/Small prefix independently of the supervised-only
branch, e.g. build it once from `("Mini" if mini else "Small" if small else "") + "TreeBoxes"
+ ("" if include_unsupervised else "_supervised")`, and drop the later unconditional
"restore." Needs the same treatment in all three loaders — this is shipped-dataset-code, so
scope and test it as its own change (per the memory note's guidance) rather than folding it
into the split fix above.

---

## 5. MEDIUM — Point OOD macro-average overweights the NEON sampling program

From `notes/ood_split_test_sources_and_leaks.md` §10b, re-verified against current
`OUT_OF_DISTRIBUTION_TEST_SOURCES_POINTS` (`package_datasets.py:46-52`): the five declared
point hold-out sources are `NEON_points`, `NEON MultiTemporal`, `OSBS megaplot 2025`,
`OFO field 2025`, `Dubrovin et al. 2024`. The first three are one sampling program at
overlapping sites (HARV, BART, WREF, ABBY, OSBS). Since the leaderboard's headline number is
an unweighted macro-mean over *sources* (`milliontrees_dataset.py:546-560`), NEON effectively
casts 3 of 5 votes. Not a leak — different imagery/years, not duplicate tiles — but it does
mean the OOD point headline is more "how well does this model do on NEON-style imagery" than
the five-source average implies. The note already records Ventura et al. 2022 (SoCal urban)
as the natural next hold-out if this needs correcting; not blocking a v1 release, but worth a
sentence in `docs/leaderboard.md` or `docs/datasets.md` so a reader of the OOD points score
knows the composition, rather than only being in the internal note.

---

## 6. MEDIUM — Module-level `version` global instead of a parameter

`create_directories`, `copy_images`, `copy_masks`, and `copy_packaged_assets_from_full`
(`package_datasets.py:619-820`) read a bare `version` name that is **not** a parameter of
`create_directories` or `copy_images`/`copy_masks` — it resolves to the module-level global
set once at `if __name__ == "__main__": version = "v0.24"` (line 1715). This only works today
because `run()`'s local `version` parameter is always called with that same literal at the
bottom of the file. If `run()` is ever imported and called with a different version (a test,
a notebook, a script that loops over versions), these functions would silently write to /
read from the **previous** global's directories rather than the one actually passed in — a
silent cross-version file corruption risk, not currently triggered but latent.

### Fix

Thread `version` through as an explicit parameter to these four functions (they already take
`base_dir` and `dataset_type`; `version` belongs alongside those), removing the implicit
global dependency. Low urgency since nothing currently calls `run()` with a mismatched
version, but worth doing before this script gets reused in the way finding #2's tests would
want (calling `run()` or its pieces programmatically in a test).

---

## 7. MEDIUM (spot-checked, not fully audited) — Trusting upstream `existing_split` without verifying spatial disjointness

Several sources ship (or are assigned, via `assign_canopyrs_aligned_existing_split`) their
own upstream train/test partition, which `apply_existing_splits` honors as-is in the
within-distribution split: `Weecology_University_Florida`, `Cloutier et al. 2023` (by
`-zN-` zone), `OAM-TCD` (by `train_`/`test_` prefix), `Troles et al. 2024` (by forest),
`NeonBenchmark` (all test), `UrbanLondon`, and — importantly — `SelvaBox`, whose tiles overlap
heavily (up to 16x per crown, per `[[project_selvabox_partial_download]]` memory).

**Spot-checked SelvaBox specifically:** the memory record shows train/valid/test are a
spatial partition by *area* (74% / 13% / 13%) with overlap only *within* each fold (tiles
inside the train region overlap other train tiles; same for test) — i.e. train and test
occupy disjoint ground regions before tiling. That's a legitimate design, not a leak. I did
not re-verify this from the raw SelvaBox parquet myself in this session; it rests on the
existing memory record's affine-transform analysis.

**Not audited:** whether Cloutier's zone boundaries, Troles' forest boundary, or OAM-TCD's
official split are actually spatially clean (no overlapping tiles straddling the boundary).
These are long-established upstream splits reused by CanopyRS itself, so they're likely fine,
but "likely fine because a well-known benchmark uses it" is an assumption, not a check.
Recommend a one-time lightweight audit (project each source's train/test tile footprints and
check for bounding-box overlap across the split) before calling the released splits verified
clean, since this is exactly the category of bug that produced findings #1 and #3.

---

## 8. LOW — Documentation understates the split-granularity guarantee

`docs/dataset_structure.md:27` says of within-distribution: "For each source, most images are
used for training and a subset for testing (the same sources appear in both splits)." This is
technically true today only at the *source* level, not the *image* level (finding #1). Once
finding #1 is fixed, update this line to state the invariant explicitly — no image appears in
more than one split — so it's a documented guarantee a user can rely on, not just an implicit
assumption.

---

## Recommended order of operations for v1

1. Fix finding #1 (image-level split) and finding #2 (add the disjointness test) together —
   the test should fail against the *current* code, proving it would have caught #1, then
   pass after the fix.
2. Fix finding #4 (mini flag) in a separate, scoped change (it touches all three loaders'
   public `__init__` signatures' behavior).
3. Repackage once, incorporating #1 and the already-coded #3 fix, bump `_versions_dict` in all
   three loaders, rerun the three leaderboard components (`CLAUDE.md` §3).
4. Add the finding #5 sentence to `docs/leaderboard.md` / `docs/datasets.md` and the finding
   #8 correction to `docs/dataset_structure.md` in the same PR as the repackage, so docs and
   data move together.
5. Findings #6 and #7 are lower urgency and can follow as hygiene/audit work; #6 especially
   should land before `run()` or its helpers are called from anywhere other than
   `__main__` (e.g. from the new test in #2, if that test ends up invoking packaging code
   directly rather than reading already-packaged CSVs).
