# TreeFormer Points: Why is recall stuck in the 20s? — Diagnosis & Experiment Plan

**Status:** open investigation (2026-06-14)
**Scope:** TreePoints only. Polygons / Mask R-CNN are tracked separately.
**Owner question (from `manuscript_table3_v0.18.csv`):**

| Geometry | Model | Fine-tuned | Split | MAE | Recall | Precision |
|---|---|---|---|---|---|---|
| Points | TreeFormer | No  | Random   | 52.94 | **0.28** | 0.85 |
| Points | TreeFormer | Yes | Random   | 57.44 | **0.28** | 0.79 |
| Points | TreeFormer | No  | Zero-shot | 18.63 | 0.39 | 0.86 |
| Points | TreeFormer | Yes | Zero-shot | 33.56 | 0.46 | 0.82 |

Two things look wrong:
1. **Fine-tuning adds ~nothing** (random recall identical 0.28→0.28; MAE *worsens* 52→57).
2. **Recall is stuck in the 20s–40s** even though the Comet training curves
   ([run 36a28c3a](https://www.comet.com/bw4sz/milliontrees-treeformer-points/36a28c3a90f549638f3b636603fff64c))
   look healthy.

The user's lead hypothesis — *"Could it be the evaluation routines?"* — is, on
inspection of the code, **very likely correct**. This document records the
mechanism, then lays out a fast iteration harness and a prioritized experiment
list so we never wait 20 h to learn something a 5-minute run can tell us.

---

## 1. The smoking gun: two different "recall" numbers

There are **two independent recall computations** in play, and they do not agree
by construction. The healthy Comet curve and the stuck leaderboard number are
*not the same metric*.

### (A) Comet curve `point_recall` — what training optimizes/checkpoints
- Computed by DeepForest's own validation metric
  (`deepforest/metrics.py::RecallPrecision`, `task="point"`).
- A predicted point is a true positive if `distance < distance_threshold`,
  where the **default `distance_threshold = 10.0` pixels** in the model's 448-px
  working resolution (`metrics.py:23,144`).
- **No score filtering.** Every peak returned by `density_to_points`
  (`utilities.py:981`) is used. The only gate is the relative peak threshold
  `score_thresh=0.1` inside `peak_local_max`.
- This is what `training/points/train.py` monitors for checkpointing and early
  stopping (`monitor="point_recall", mode="max"`, `train.py:268-281`). It looks good.

### (B) Leaderboard `KeypointAccuracy` — what the manuscript table reports
Computed by `TreePoints.eval` → `KeypointAccuracy` (`all_metrics.py:681`). It
differs from (A) in **three** ways, each of which independently *depresses* recall:

1. **A hard score filter at `eval_score_threshold = 0.4`** (`TreePoints.py:111`,
   applied at `all_metrics.py:717`: `pred[scores > 0.4]`).
   **The scores are not calibrated confidences.** They are *per-image
   max-normalized density-map heights* (`utilities.py:1009-1027`):
   the single global-max peak in each image always scores `1.0`, and every
   other peak scores its density value *relative to that max*. In a dense
   canopy with many comparable peaks, most true detections land between 0.1 and
   0.4 → **they are thrown away before matching.** This alone can cap recall in
   the 20s regardless of how good the model is.
2. **A different, GSD-derived distance threshold.** `TreePoints.eval` recomputes
   a per-source radius `real_world_3m / (gsd · orig_px)` then scales by
   `image_size` (`TreePoints.py:358-382`, `all_metrics.py:734`). This is a
   different matching radius than DeepForest's flat 10 px — sometimes tighter.
3. **Greedy 1:1 matching** (`matching.py::greedy_distance_match`) vs DeepForest's
   per-prediction nearest-GT rule. 1:1 is stricter (no GT double-counting).

> **Net:** the leaderboard recall is gated primarily by a 0.4 threshold applied
> to a quantity that isn't a probability. This is the first thing to falsify.

### Why fine-tuning "does nothing"
If the 0.4 filter is the binding constraint, both the pretrained and fine-tuned
models are scored through the same lossy gate, so genuine model improvements are
masked. The random-split MAE getting *worse* (52→57) while recall is pinned is
consistent with "the metric, not the model, is saturating."

---

## 2. Fast iteration harness (the proxy for 20-hour runs)

We do **not** need to retrain to answer the eval question, and we never need the
full release to iterate.

### Levers we already have
- **`SmallTreePoints_v0.18`** exists on disk
  (`/orange/ewhite/web/public/MillionTrees/SmallTreePoints_v0.18`, ~72 k points,
  all the key sources represented with train/test). This is the iteration set.
- **Single-inference threshold sweep** is already built:
  `existing_models/treeformer/eval_points.py` imports `add_sweep_args` /
  `maybe_run_sweep` (`src/milliontrees/common/eval_sweep.py`). It runs inference
  **once** and re-scores `KeypointAccuracy` across a grid of `eval_score_threshold`
  by mutating the live metric objects — so a full recall-vs-threshold curve costs
  one forward pass, not one run per threshold.
- **`maybe_subsample` / `--max-batches`** cap the test set for sub-minute smoke runs.

### Small gaps to close first (≤30 min of plumbing, do these as step 0)
- `eval_points.py` plumbs `--mini` but **not** `--small` or `--image-size` to
  `get_dataset`. Add both flags (the dataset already accepts `small=` and
  `image_size=`). Mirror in `training/points/eval.py`.
- Expose `--eval-score-threshold` on the eval scripts (today it's hard-wired via
  the dataset default 0.4) so the sweep can start at ~0.

### Proxy contract
Every experiment below reports on **`--small --split-scheme random`** first.
Promote to the full release **only** after a hypothesis clears its decision gate
on small. Record each run in `/home/b.weinstein/logs/job_ledger.md` per CLAUDE.md.

---

## 3. Experiment list (ordered; stop early when a gate fails)

### Phase 0 — Reproduce the discrepancy cheaply  *(no training)*
**E0. Recompute both recalls on the same predictions.**
Run pretrained TreeFormer once on `--small random`, dump predictions, and score
with (a) DeepForest's `RecallPrecision` (flat 10 px, no score filter) and
(b) `TreePoints.eval`. Confirm (a) ≈ Comet curve and (b) ≈ 0.28.
- *Gate:* if the two numbers reproduce the gap, the eval routine is confirmed as
  the dominant factor and Phase 1 is the priority. If they *agree*, the model
  really is weak → jump to Phase 3.

### Phase 1 — Fix/justify the eval routine  *(no training; highest expected value)*
**E1. `eval_score_threshold` sweep** via `maybe_run_sweep`, grid e.g.
`0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5`. Plot recall/precision/F1 vs threshold per
source. *Expectation:* recall climbs steeply as the threshold drops below 0.4.
- *Gate:* if recall at threshold≈0.1 jumps materially (e.g. 0.28→0.5+) at
  acceptable precision, the leaderboard threshold is mis-set → pick a principled
  default and re-score **all** point baselines consistently.

**E2. Score-semantics fix.** Decide what "score" should mean for a density model.
Options to test (cheap, post-hoc on dumped density maps):
  - Use *raw* (un-normalized) density peak height, or a global (dataset-level)
    normalization, instead of per-image max-normalization, so 0.4 means the same
    thing across images.
  - Or drop the score filter for points entirely and rely on
    `score_integration_radius` (peak spacing) for precision control.
- *Gate:* a scoring scheme where a single fixed threshold gives stable
  precision/recall *across sources*.

**E3. Distance-threshold reconciliation.** Compare GSD-derived radius vs flat
pixel radius per source; verify the GSD/`orig_px` values are right (the
`SOURCE_GSD` table drives the radius). A wrong GSD on a big source silently
tanks its recall.
- *Gate:* matching radius corresponds to a sane real-world distance (≈3 m) for
  every evaluated source.

### Phase 2 — Post-processing sweeps  *(no training)*
**E4. `score_integration_radius` (peak_local_max `min_distance`) sweep**
`{1,2,3,5}`. Lower → more, closer peaks (recall↑, merge-commission↓ risk).
Already flagged as the dominant recall lever in memory
(`project_treeformer_recall_lever`). Sweep jointly with E1.

**E5. `image_size` sweep** `{448, 672, 896, 1280}`. Density grid is
`image_size/4` per side; larger → finer separation of close trees in dense
canopy → recall↑. Needs the `--image-size` plumbing from step 0. Inference-time
only for the pretrained baseline.
- *Gate (Phase 2):* best (st, radius, image_size) combo on small that maximizes
  F1 — freeze it as the standardized post-proc for all point runs.

### Phase 3 — Data / training experiments  *(training, only if Phases 1–2 don't close the gap)*
**E6. Per-source train ablation.** Some point sources are weak supervision /
loosely aligned (e.g. `Beery et al. 2022` is already eval-demoted as train-only,
`TreePoints.py:67`). Train on curated subsets (drop suspected-buggy sources) and
compare. Use `include_sources` / `exclude_sources` (already supported).
**E7. LR / init re-check** *after* the metric is trustworthy — the "fine-tuning
does nothing" claim must be re-evaluated against a fixed metric, since E1 may
explain it entirely. Compare pretrained-KCL vs fine-tuned vs scratch on small.
**E8. KCL-weights provenance note.** "No / pretrained" = the
`weecology/deepforest-tree-point` checkpoint, trained on the Amirkolaee (KCL)
source. If that source dominates the *test* distribution, zero-shot's higher
recall (0.39) vs random (0.28) may be a source-mix artifact, not a real
generalization story — quantify per-source.

---

## 4. Decision flow

```
E0 ─ gap reproduces? ──no──► model is genuinely weak → Phase 3 (E6/E7)
       │ yes
       ▼
E1/E2/E3 ─ recall recovers at a principled threshold/score? ──yes──► restandardize
       │                                                              eval + re-score
       │ no                                                           ALL baselines
       ▼
E4/E5 ─ post-proc recovers recall? ──yes──► freeze post-proc, re-score
       │ no
       ▼
Phase 3: data curation + training (only now is it worth a 20 h run)
```

---

## RESULTS (2026-06-15, SmallTreePoints v0.18, random, pretrained TreeFormer)

### E0/E1 — score threshold is NOT the culprit (hypothesis falsified)
Single-inference `eval_score_threshold` sweep (job 34671792). Dropping the filter
from 0.40 → 0.05 moved recall only **0.242 → 0.274**. Recall is capped ~0.27 even
with no score filtering, at high precision (0.83–0.95). ⇒ **under-detection**, not
over-filtering. Phase 1 (E1/E2) is closed.

### E5 — `image_size` is THE lever (hypothesis confirmed)
Grid (job 34694394), `integration_radius=2`, matching radius held at a constant
3 m (the GSD threshold scales with `image_size`, so the gain is real, not an
artifact):

| image_size | density grid | avg recall | avg precision | avg counting MAE |
|---|---|---|---|---|
| 448  | 112 px | 0.27 | 0.89 | 110 |
| 672  | 168 px | 0.42 | 0.88 | 70  |
| 896  | 224 px | **0.55** | 0.81 | **60** |
| 1280 | 320 px | 0.73 | 0.21 (worst-grp) | 396 (over-fires) |

Every source roughly doubles from 448→896. `integration_radius=1` vs `2` is a
minor (~+0.01 recall) effect. **896 is the single-value sweet spot**: recall
doubles, precision stays high, MAE halves. 1280 over-detects (precision/MAE
collapse).

**The v0.18 leaderboard point numbers were generated at `image_size=448`, which
under-resolves dense canopy.** Re-scoring at 896 ≈ doubles recall.

### E3 — Bohlman is a separately broken source
`Bohlman et al. 2008` recall ≈ 0 at every resolution (0.007 → 0.004 → 0.021),
with the tightest matching radius of all sources (0.0067 norm; gsd=0.30). Needs
its own GSD/`orig_px`/annotation audit — a global resolution change won't fix it.
It drags the average and is the worst-group at every size.

### Revised conclusion
The "stuck in the 20s" recall is **eval resolution**, not the 0.4 score filter and
not (primarily) the GSD radius. The "fine-tuning does nothing" claim must be
re-tested at 896 — at 448 both models were resolution-starved, so any training
gain was invisible.

### Full-release confirmation (2026-06-15, job 34699513) — recall win, MAE caveat
Pretrained TreeFormer, full TreePoints v0.18, 448 (control) vs 896:

| split | size | recall | precision | counting MAE |
|---|---|---|---|---|
| random   | 448 | 0.284 | 0.847 | **52.9** |
| random   | 896 | **0.453** | 0.795 | 126.6 ⚠️ |
| zeroshot | 448 | 0.392 | 0.857 | 18.6 |
| zeroshot | 896 | _(running)_ | | |

- **Recall win confirmed at scale**: 0.284 → 0.453 on random, every source up.
- zeroshot/448 reproduces the leaderboard exactly (recall 0.39, MAE 18.63) ⇒ the
  small/eval harness is faithful.
- **MAE caveat (new):** counting MAE *regressed* at 896 on the full release
  (53 → 127), the OPPOSITE of the small set (110 → 60). The small subsample
  over-weighted dense sources where higher res helps counting; the full release
  includes sparser images where 896 over-fires and inflates counts. **The small
  set is a faithful proxy for recall but NOT for counting MAE.**
- ⇒ Do **not** blindly swap the leaderboard to 896. 896 trades counting for
  detection. **672** is the suspected balance (small: recall 0.42, MAE 70, both
  beat 448); full-release 672 running as job 34701796 to decide.

### Full image_size tradeoff (2026-06-15, jobs 34699513 + 34701796) — DECISION TABLE
Pretrained TreeFormer, full TreePoints v0.18, `score_thresh=0.1`, `radius=2`:

| split | size | recall | precision | counting MAE |
|---|---|---|---|---|
| random   | 448 | 0.284 | 0.847 | **52.9** |
| random   | 672 | 0.388 | 0.823 | 68.0 |
| random   | 896 | **0.453** | 0.795 | 126.6 |
| zeroshot | 448 | 0.392 | 0.857 | **18.6** |
| zeroshot | 672 | 0.484 | 0.838 | 41.4 |
| zeroshot | 896 | **0.527** | 0.809 | 107.3 |

Monotone in both splits: recall↑, precision↓ (slightly), MAE↑ with resolution —
a clean recall-vs-counting tradeoff, no free lunch on the full distribution.

**Recommendation: 672 as the single leaderboard resolution.** It recovers most of
the recall gain (+37% random, +23% zeroshot) at a bounded MAE cost (53→68,
19→41), whereas 896 roughly doubles MAE again (→127, →107) for a smaller
marginal recall gain. Precision barely moves. (448 stays MAE-best if the
benchmark prioritizes counting over detection — a values call for the PI.)

### E7 — fine-tuning DOES help; "no value" was a stale-table artifact (jobs 34699513 vs 34709163)
Apples-to-apples (same eval pipeline, same `score_thresh=0.1`/`radius=2`),
pretrained vs fine-tuned TreeFormer, full release:

| split | size | pretrained recall | fine-tuned recall | Δ | FT MAE |
|---|---|---|---|---|---|
| random   | 448 | 0.284 | 0.366 | +0.082 | 65.9 |
| random   | 672 | 0.388 | **0.514** | +0.126 | 148.0 |
| zeroshot | 448 | 0.392 | 0.459 | +0.067 | 33.6 |
| zeroshot | 672 | 0.484 | _(pending 34709163_1)_ | | |

- **Fine-tuning improves recall in every comparison**, and the gain is *larger*
  at 672 (+0.13) than 448 (+0.08) ⇒ resolution and fine-tuning are
  **complementary**, not redundant.
- **Why the table said "no value":** the v0.18 `Yes/Random` row (0.28, MAE 57.44)
  is **stale** — from an older/weaker random checkpoint. The current Jun-15
  checkpoint scores 0.353 (train.py end-of-run) / 0.366 (E7 re-eval) at 448. The
  `Yes/Zeroshot` row (0.46) *was* current (E7 zeroshot@448 reproduces 0.459
  exactly). My pretrained@448 also reproduces the `No` rows exactly ⇒ the eval
  pipeline is consistent; only the random FT row was outdated.
- **MAE caveat persists:** FT over-counts more than pretrained (random@672 MAE
  148 vs 68); fine-tuning trades counting for recall, same direction as resolution.
- **Train/eval mismatch caveat:** these checkpoints were *trained* at 448, so
  eval@672 is off-distribution for the density head. A clean test needs a
  **retrain at 672** — expected to push FT recall higher still and possibly
  temper the MAE blow-up. This is the next training job (not just inference).

#### Final manuscript numbers — 672, Bohlman-clean (jobs 34733817 pretrained / 34733818 finetuned)
The E7 table above predates the Bohlman 2008 removal (and used the best-`point_recall`
ckpt). For the manuscript we re-ran both arms @672 with Bohlman dropped from the config
and the **lowest-`val_loss`** checkpoint (leaderboard convention, matches
`eval_points.sbatch`). These are the numbers in `notes/manuscript_table3_v0.18.csv`:

| split | pretrained recall / prec / MAE | fine-tuned recall / prec / MAE | Δ recall |
|---|---|---|---|
| random   | 0.436 / 0.823 / 67.95 | **0.577** / 0.780 / 150.97 | +0.141 |
| zeroshot | 0.484 / 0.838 / 41.36 | **0.599** / 0.764 / 161.11 | +0.115 |

- Conclusion holds and strengthens: **fine-tuning lifts recall by +0.12–0.14** at the
  leaderboard resolution, on the clean source set, with the leaderboard checkpoint.
- The **MAE blow-up is now the headline cost**: FT roughly doubles (random 68→151) to
  quadruples (zeroshot 41→161) counting MAE, and trims precision ~0.04–0.07. Fine-tuning
  buys recall by over-firing the density head — same direction as raising resolution.
- Train/eval mismatch (ckpts trained @448) still applies; a **retrain at 672** remains
  the clean follow-up and is expected to temper the MAE cost.

### Next actions (revised)
1. **Re-score ALL point baselines at `image_size=896`** (pretrained + fine-tuned,
   random + zeroshot, SAM3) → corrected leaderboard. Promote the small-set finding
   to the full release first to confirm it holds.
2. **Per-source optimal resolution** (optional): recall is still source-dependent;
   matching eval resolution to each source's GSD/tree spacing may beat a single
   global 896.
3. **Audit Bohlman** (GSD, native px, annotation density) — E3/E6.
4. **Re-open the fine-tuning question (E7) at 896** — and retrain at 896 (train
   default is also 448), since a finer density target may change what fine-tuning
   can learn.

---

## 5. First concrete actions
1. Step 0 plumbing: add `--small`, `--image-size`, `--eval-score-threshold` to
   `existing_models/treeformer/eval_points.py` and `training/points/eval.py`.
2. **E0 + E1** in one job: pretrained TreeFormer on `--small random`, with the
   `--sweep` threshold grid. ~minutes on one GPU. Ledger it.
3. Read the per-source sweep table; pick the gate outcome; branch per §4.

**Do not** launch a full-release retrain until E0–E5 have a verdict.
