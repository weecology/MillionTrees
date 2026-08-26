# Does the supervised set swamp the weak-supervision benefit? (box → box, v0.23 / AP40)

> ## ⚠️ CONCLUSION SUSPENDED — 2026-08-21
>
> This grid tested transfer from a backbone that the audit in
> `notes/weak_supervision_stage1_audit.md` shows is **undertrained** (22% recall on its own
> training labels; constant LR; last-epoch export with no checkpoint selection), and every
> cell then fine-tuned at **100× the pretraining learning rate with no warmup**.
>
> The "dataset size is not the mechanism" conclusion below does not follow until stage 1
> is re-run to convergence. The aggregation critique (macro-average over 16 sources is
> dominated by n=1 sources) stands on its own and is still correct.



*Opened 2026-08-19. Companion to `notes/weak_supervision_pretraining_table.md`.
Results pending — job `39708566`.*

## The problem this run exists to solve

Every version of the weak-supervision ablation has come back null, and the last loophole
just closed. Transferring the **entire** 301-tensor pretrained RetinaNet — a model that is
already a working tree detector before fine-tuning starts — reproduces the COCO baseline
exactly:

| Split | COCO control | Whole-network transfer | Δ AP40 |
|---|---:|---:|---:|
| within-distribution | 0.476 | 0.476 | 0.000 |
| out-of-distribution | 0.517 | 0.501 | -0.016 |

It is hard to believe 6.4M unsupervised boxes (41,691 images) encode *nothing* useful. The
more likely reading is that the measurement has no headroom: fine-tuning on 13,538 images
and 796,496 human-quality boxes reaches the same solution from any sane initialization, so
whatever pretraining contributed is overwritten long before the last epoch.

## The test

Shrink the **supervised** training set, hold everything else fixed, and watch whether the
pretraining delta grows as supervision shrinks.

- `--train-sources` (new, `TreeBoxes(train_sources=...)`) filters the **TRAIN split only**.
  Validation and test are untouched, so every cell below is scored on the same **3,023
  test images** as the full-data baselines and the numbers are directly comparable.
- `--train-frac` subsamples train images, seeded, so all three inits in a tier see the
  identical image subset.
- `--val-frac 0.25` shrinks only the per-epoch monitoring metric that drives early
  stopping; the final scored eval always runs on the full test split.

### Grid (within-distribution split, job `39708566`)

| Tier | Train sources | Train images | Train boxes | Domain vs. pretraining data |
|---|---|---:|---:|---|
| `weecology10` | Weecology_University_Florida, 10% | ~206 | ~4.9k | matched (NEON) |
| `selvabox` | SelvaBox | 915 | 266,383 | mismatched (tropical) |
| `weecology` | Weecology_University_Florida | 2,056 | 48,969 | matched (NEON) |
| *(full, done)* | all 15 sources | 13,538 | 796,496 | mixed |

× three inits: `coco` (control), `box_pretrained` (trunk, 265/301 tensors),
`box_pretrained_full` (whole network, 301/301).

Same recipe as the full-data arms (bs 32, lr 0.01, grad-clip 1.0, early stop on
`box_recall`). Epoch budget and patience scale per tier so each cell gets a comparable
number of optimizer steps — patience counts epochs, and a 206-image epoch is 7 steps:
400/40, 300/30, 200/20 respectively.

## How to read the outcome

- **Δ grows as train size falls** → the null at full data is a headroom artifact. The weak
  labels do carry signal; MillionTrees is simply large enough to make it redundant. That is
  a publishable, interpretable result and it changes what Table 6 should claim.
- **Δ stays flat at ~206 images, including the whole-network arm** → dataset size is *not*
  the mechanism. A model that begins as a tree detector and gains nothing on 206 images
  means the pretraining or the transfer is broken, and the pipeline is what needs auditing
  next (candidate suspects: unsupervised box quality, the 20-epoch/lr-1e-4 pretraining
  recipe, BN statistics).
- **`selvabox` vs `weecology` at similar size** separates "small data" from "domain match":
  the pretraining corpus is NEON temperate imagery, so a benefit that appears only in the
  `weecology` tier is domain transfer, not general feature learning.

## Results

*Job `39708566`, all 9 tasks COMPLETED 0:0 on 2026-08-19. Train-image counts confirmed in every
log (206 / 914 / 2,056); init banners verified in all 9 cells. Scored on the full 3,023-image
within-distribution test split, so every row is comparable to the full-data baselines.*

### The grid, as reported (`Average AP40` = macro-average over 16 test sources)

| Tier | Train images | Init | Recall | Mask-aware precision | AP40 | Δ AP40 vs COCO |
|---|---:|---|---:|---:|---:|---:|
| `weecology10` | 206 | COCO (control) | 0.434 | 0.572 | 0.256 | — |
| `weecology10` | 206 | weak trunk (265) | 0.336 | 0.420 | 0.135 | **−0.121** |
| `weecology10` | 206 | weak whole-net (301) | 0.437 | 0.516 | 0.273 | +0.017 |
| `selvabox` | 914 | COCO (control) | 0.445 | 0.500 | 0.236 | — |
| `selvabox` | 914 | weak trunk (265) | 0.339 | 0.469 | 0.137 | **−0.099** |
| `selvabox` | 914 | weak whole-net (301) | 0.381 | 0.461 | 0.197 | −0.039 |
| `weecology` | 2,056 | COCO (control) | 0.462 | 0.549 | 0.274 | — |
| `weecology` | 2,056 | weak trunk (265) | 0.429 | 0.477 | 0.261 | −0.013 |
| `weecology` | 2,056 | weak whole-net (301) | 0.484 | 0.512 | 0.310 | +0.036 |
| *(full)* | 13,538 | COCO (control) | 0.637 | 0.633 | 0.476 | — |
| *(full)* | 13,538 | weak trunk (265) | 0.635 | 0.642 | 0.460 | −0.016 |
| *(full)* | 13,538 | weak whole-net (301) | 0.631 | 0.644 | 0.476 | 0.000 |

### The macro-average is not trustworthy at this scale — check it three ways

`Average AP40` weights all 16 test sources equally, and three of them have n ≤ 12 (Kwon n=1,
Šrollerů n=1, Sun n=12). A single image can therefore move the headline by 1/16 of its own swing.
It does:

- At 206 images, Šrollerů (**n = 1**) goes 0.265 → 0.467 under the whole-network init, contributing
  **+0.013 of the +0.017** headline delta — roughly three quarters of it, from one image.
- At 2,056 images, Šrollerů (n=1, +0.250) and Zamboni (n=24, +0.159) together contribute +0.025 of
  the +0.036 — about 70%, from 25 images.

Recomputing the same runs under two more defensible aggregations:

| Tier | Train images | Init | macro (16 sources) | macro (n ≥ 20 only) | image-weighted |
|---|---:|---|---:|---:|---:|
| `weecology10` | 206 | COCO | 0.256 | 0.307 | 0.260 |
| `weecology10` | 206 | weak trunk | 0.135 | 0.167 | 0.150 |
| `weecology10` | 206 | weak whole-net | 0.273 | 0.311 | 0.249 |
| `selvabox` | 914 | COCO | 0.236 | 0.281 | 0.298 |
| `selvabox` | 914 | weak trunk | 0.137 | 0.162 | 0.196 |
| `selvabox` | 914 | weak whole-net | 0.197 | 0.241 | 0.262 |
| `weecology` | 2,056 | COCO | 0.274 | 0.324 | 0.314 |
| `weecology` | 2,056 | weak trunk | 0.261 | 0.300 | 0.269 |
| `weecology` | 2,056 | weak whole-net | 0.310 | 0.356 | 0.309 |
| *(full)* | 13,538 | COCO | 0.476 | 0.539 | 0.447 |
| *(full)* | 13,538 | weak trunk | 0.460 | 0.529 | 0.430 |
| *(full)* | 13,538 | weak whole-net | 0.476 | 0.538 | 0.450 |

**Whole-network Δ AP40 vs COCO, by aggregation:**

| Train images | macro (16) | macro (n ≥ 20) | image-weighted |
|---:|---:|---:|---:|
| 206 | +0.017 | +0.004 | −0.011 |
| 914 (domain-mismatched) | −0.039 | −0.040 | −0.036 |
| 2,056 | +0.036 | +0.032 | −0.005 |
| 13,538 | 0.000 | −0.001 | +0.003 |

No aggregation shows Δ growing as supervision shrinks, and the only tier that looks positive under
two of the three (`weecology`, 2,056 images) is flat under the third. The sign is not stable, the
trend is absent, and the magnitude never exceeds the run-to-run noise the ladder in
`notes/weak_supervision_pretraining_table.md` section 1 already demonstrated.

## Conclusion: dataset size is NOT the mechanism

Applying this note's own decision rule (see "How to read the outcome" above): **Δ stays flat at 206
images, including the whole-network arm.** Shrinking supervision 65× does not open a gap for a
model that begins life as a complete, verified tree detector. The null at full data is therefore
not a headroom artifact of MillionTrees being large.

Combined with the ImageNet rung (job `39708673`, added to section 1 of the pretraining note), which
shows that removing detection pretraining *entirely* costs only 0.020 AP40 within-distribution,
the picture is consistent: **initialization contributes very little to this task at any of the
supervision levels tested, and weak-label pretraining contributes nothing beyond COCO.**

### What the grid *did* establish

1. **Damage, not benefit, is what large supervision erases.** The trunk-only init — which resets
   the FPN and both detection heads — costs **−0.121 AP40 at 206 images**, −0.099 at 914, −0.013
   at 2,056 and −0.016 at full data. Initialization clearly matters when data is scarce; it is
   specifically the *weak-label* content that adds nothing. Note also that the trunk-only cells ran
   far longer before early stopping (367 epochs at the 206-image tier vs 101 for COCO and 90 for
   whole-network) and still finished worst — this is not an under-training artifact.
2. **Domain match does not rescue it.** The pretraining corpus is NEON temperate imagery, so the
   `weecology` tier is domain-matched and `selvabox` is not. The matched tier is the less negative
   of the two, but the per-source breakdown does not support a domain-transfer story: at the
   `weecology` tier the *NEON* test sources actually move **down** under the whole-network init
   (NEON_benchmark −0.018, Weecology_University_Florida −0.011, NEON MultiTemporal −0.037) while
   the gains land on unrelated sources (Kaggle_Palm +0.057, Radogoshi +0.062, Velasquez +0.077).
   That is the signature of noise, not of transferred NEON features.
3. **SelvaBox is hurt in every tier by every pretrained init.** The largest test source (n=1,535)
   drops under both treatments at all three tiers (whole-net: −0.035 / −0.047 / −0.037). This is
   the most consistent single signal in the grid and it is negative.

## What to run next

The grid rules out volume, so the remaining suspect is stage 1 itself — and **stage-1 quality has
never been measured.** `training/weak_supervision/outputs/pretrain_*/<split>/results_<split>.txt`
are all `nan` because that stage is evaluated on the unlabeled pretraining split. The decisive,
cheap (~0.5 h) test is a **zero-shot eval of `box_network_<split>.pt`** (301 tensors, 129 MB,
verified on disk) against the standard box test split:

- **Zero-shot AP40 ≈ 0.3+** → the weak-label features are real, and genuinely redundant with what
  supervision already provides. The null is a headroom result about MillionTrees, and publishable
  as one.
- **Zero-shot AP40 ≈ 0** → stage 1 never produced a tree detector, and every null in these two
  notes is measuring a broken pretraining stage rather than a saturated benchmark. Audit the
  unsupervised box quality, the 20-epoch / lr-1e-4 recipe, and the BN statistics next.

Optional second run: add an `imagenet` arm to this grid. Finding 1 predicts a large ImageNet−COCO
gap at 206 images (where the ladder's full-data gap is only 0.020), which would confirm that the
grid is sensitive enough to detect an initialization effect when one exists — the control this
null currently lacks.
