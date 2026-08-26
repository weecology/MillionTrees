# Plan: measuring whether weak pretraining helps, conclusively

*2026-08-21. Supersedes the ad-hoc arms in `notes/weak_supervision_pretraining_table.md`.
Background: `notes/weak_supervision_stage1_audit.md`.*

**The question.** Does pretraining on the 7.1M weak boxes (Weinstein et al. 2018, 46,823
NEON tiles, 25 sites) improve performance on the **supervised** test and validation splits?

Two independent experiments, reported as two tables:

- **A — Box → Box.** Does a box model pretrained on weak boxes beat one trained from COCO?
- **B — Box → Polygon (Table 6).** Does a polygon model with a weak-pretrained backbone beat
  one with a COCO backbone?

## Why the current arms cannot answer it

Three confounds, each large enough to produce the observed null on its own:

| # | Confound | Affects | Fix |
|---|---|---|---|
| 1 | Stage 1 is undertrained — val_loss barely moves, supervised `box_recall` is **0.0000** after 3 epochs of the original recipe | A and B | train to convergence, verified against held-out supervised images |
| 2 | Stage 2 fine-tunes at **100×** the stage-1 LR, no warmup, no schedule | A and B | one identical stage-2 recipe for every arm, with warmup |
| 3 | The polygon transplant grafts a **RetinaNet** backbone into **Mask R-CNN v2** — different architecture (FrozenBN vs BatchNorm), different weights (0/265 tensors shared) | B only | pretrain in the polygon model's own architecture (below) |

Confound 3 is the one that has no fix by adding a control arm — a "RetinaNet-COCO trunk
transplant" control would only measure damage we should not be incurring at all.

## The fix for confound 3: pretrain in the right architecture

`fasterrcnn_resnet50_fpn_v2` and `maskrcnn_resnet50_fpn_v2` are **architecturally
identical everywhere except the mask head**:

| Group | Tensors in Mask R-CNN v2 | Fillable from a trained Faster R-CNN v2 |
|---|---:|---:|
| `backbone.body` | 318 | **318** |
| `backbone.fpn` | 48 | **48** |
| `rpn` | 8 | **8** |
| `roi_heads.box` | 30 | **30** |
| `roi_heads.mask` | 28 | 0 (stays at COCO) |
| **total** | **432** | **404** |

So stage 1 for experiment B becomes: build a **Faster R-CNN v2**, initialise it from
**Mask R-CNN v2's own COCO weights**, train it on the weak boxes, then load those 404
tensors back into Mask R-CNN v2.

The control is then *literally stage 1 at epoch 0*. Nothing differs between arms except
what the weak boxes taught. No architecture mismatch, no BatchNorm-type mismatch, no
foreign weights.

## Recipe decisions, from the LR sweep (job `39879762`, 3 epochs)

| Arm | supervised `val_loss` e0/e1/e2 | supervised `box_recall` |
|---|---|---|
| lr 1e-4 constant (**the original recipe**) | 1.791 / 1.736 / 1.749 | **0.0000 every epoch** |
| lr 1e-3 cosine | **1.422 / 1.488 / 1.435** | 0.079 at e1 |
| lr 1e-2 cosine | **NaN from e0 — diverged** | – |

- The original recipe detects **nothing** on supervised images. It is not a marginal
  underperformer; it never left the ground.
- lr 1e-3 sits ~0.35 lower on val_loss within 3 epochs. **This is the recipe.**
- lr 1e-2 diverged because the pretraining stage had **no gradient clipping**, while stage 2
  has always passed `--clip 1.0`. Now added (`--grad-clip`, default 1.0).

**Throughput**: GPU utilisation is only **13–22%** at `--num-workers 4` — stage 1 is I/O
bound, not compute bound. Raising to 16 workers / 16 CPUs should cut ~986 s/epoch to
roughly 300 s, which is what makes a long run affordable.

## The runs

### Phase 1 — converged stage 1 (the "much longer" run)

Both architectures, both splits, 4 jobs in parallel. lr 1e-3, cosine, `--grad-clip 1.0`,
batch 32, 16 workers, validating every epoch on 512 held-out **supervised train** images,
exporting the best epoch by supervised `val_loss`.

| Job | Architecture | Feeds | Epochs |
|---|---|---|---:|
| 1a | RetinaNet (DeepForest) | experiment A | 60, early-stop patience 12 |
| 1b | Faster R-CNN v2, init from Mask R-CNN v2 COCO | experiment B | 60, early-stop patience 12 |

**Dose–response**: also export checkpoints at fixed budgets (epochs 5 / 15 / 30 / best).
Running stage 2 from each turns a single-point comparison into a curve — *does more weak
pretraining monotonically help?* That is what makes the answer conclusive rather than
another isolated null.

**Gate before Phase 3/4**: supervised `val_loss` must fall materially below the lr-1e-4
arm, and zero-shot supervised AP40 must be clearly above 0. If stage 1 still cannot detect
a tree on supervised imagery after 60 converged epochs, the pseudo-labels themselves become
the suspect and no downstream table is worth running.

### Phase 2 — a stage-2 recipe that does not forget

Add warmup + cosine to `training/boxes/train.py` and `training/polygons/train.py`, applied
**identically to every arm including the controls** (a recipe change that only helps the
treatment arm would be worthless). Verify on the control that it does not regress the
existing baseline (box WD AP40 0.476).

### Phase 3 — Table B (boxes)

Per split (within-distribution, out-of-distribution) × per eval split (**test** and
**validation**):

| Init | What it isolates |
|---|---|
| ImageNet | floor — no detection pretraining at all |
| COCO | control |
| COCO + weak **trunk** (265) | backbone-only transfer |
| COCO + weak **whole network** (301) | full transfer |

### Phase 4 — Table 6 (polygons), rebuilt

| Init | What it isolates |
|---|---|
| Mask R-CNN v2 COCO | control (= stage 1 at epoch 0) |
| + weak **backbone** (366: body + FPN) | "just the backbone pretrained" |
| + weak **backbone + RPN + box head** (404) | everything transferable |

## Cost

| Phase | Jobs | ~GPU-h | Wall (parallel) |
|---|---:|---:|---:|
| 1 — stage 1 | 4 | ~40 | ~12 h |
| 2 — stage-2 recipe check | 2–4 | ~25 | ~10 h |
| 3 — box table | 8 | ~20 | ~3 h |
| 4 — polygon table | 6 | ~60 | ~10 h |
| dose–response (boxes) | 6 | ~15 | ~3 h |
| validation-split evals | 14 | ~5 | ~1 h |

Roughly **2–3 days wall** with parallel submission.

## New code required

1. `training/boxes/pretrain_fasterrcnn_v2.py` — stage 1 in the polygon architecture, with
   an `assert_maskrcnn_v2_init()` guard mirroring `assert_coco_trunk()`, and a 404-tensor
   export. *(new)*
2. `--init-mode` options in `training/polygons/train.py` for the 366- and 404-tensor loads,
   strict, refusing partial cover. *(extension)*
3. Warmup/cosine plumbing in both `train.py`s. *(extension)*
4. Periodic-checkpoint export for the dose–response curve. *(small)*
