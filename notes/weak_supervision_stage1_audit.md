# Audit: the weak-supervision null is an artifact of an unmeasured, undertrained stage 1

*2026-08-21. Triggered by a challenge to the null result reported in
`notes/weak_supervision_pretraining_table.md` and `notes/weak_supervision_data_scaling.md`.*

**Verdict: the null does not support the claim "6.4M unsupervised boxes add nothing."**
Stage 1 was never validated, never model-selected, never LR-scheduled, and is badly
undertrained. The conclusions in both notes are **suspended** pending a corrected stage 1.

The data loader is *not* the problem — it checks out on every axis tested. The problem is
the pretraining recipe and the complete absence of instrumentation around it.

---

## 1. Stage 1 had no validation set at all — so nothing was watching

`pretrain_backbone_for_polygons.py` selects `include_sources=['*unsupervised*']`. Every
unsupervised row in TreeBoxes v0.23 is in the **train** split, so the run reports:

```
[MillionTrees] Split: out-of-distribution | Images train/test/total: 46823/0/46823
[MillionTrees] Annotations: 7126326 | Sources selected/available: 1/20
```

`test_subset` is empty, so `has_val = False`, and four things follow silently:

| Consequence | Mechanism |
|---|---|
| No checkpoint selection | `ModelCheckpoint` was only added `if has_val` — the **last** epoch is exported |
| No early stopping | same gate |
| **No LR schedule** | `deepforest.main.configure_optimizers` attaches a scheduler only when `config.validation.csv_file is not None`; otherwise it returns a bare optimizer. LR was **constant** |
| All-`nan` results files | the final `evaluate()` scored the empty test subset — which is why every `pretrain_*/results_<split>.txt` is `nan` |

There was therefore **no signal of any kind** that stage 1 was working. The `.out` logs
contain no loss line either: DeepForest logs `train_loss` via `self.log(...)` without
`prog_bar=True`, so the progress bar shows only `v_num`.

## 2. The loss plateaus, and the exported model is not the best one

Recovered from Comet (project `milliontrees-pretrain`), `train_loss_epoch`:

| Run | e0 | e5 | e7 | e10 | e15 | e19 |
|---|---:|---:|---:|---:|---:|---:|
| WD whole-net (`41a5e191`, job 39193929_0) | 1.663 | 1.124 | 1.077 | 1.235 | 1.043 | **1.069** |
| OOD whole-net (`5c6a2f0d`, job 39614225_1) | 1.617 | 1.212 | **1.122** | 1.268 | 1.166 | **1.217** |

Two things are wrong here:

**The loss stops improving around epoch 5 and then oscillates** for fifteen more epochs —
the signature of a constant LR that is never annealed (finding 1).

**The OOD run exported a model materially worse than its own best epoch.** Its
classification loss reached 0.529 at epoch 7 and ended at 0.651; total loss ended at
1.217 against a best of 1.122. With no `ModelCheckpoint`, the last epoch was exported
anyway. The OOD arm of the whole-network box ablation was built on those weights.

**Box regression barely trained at all.** Over 20 epochs on 7.1M boxes:

| Run | `train_bbox_regression` e0 → e19 | change |
|---|---|---:|
| WD | 0.6494 → 0.5508 | −15% |
| OOD | 0.6455 → 0.5661 | −12% |

Localization — the thing the ablation hoped would transfer — moved by roughly a tenth.

## 3. Zero-shot: a real detector, but nowhere near converged

Scoring the exported 301-tensor network directly (`scratchpad/diag_zeroshot.py`, IoU 0.4,
CPU). First against COCO init, at DeepForest's inference defaults:

| Model | Data | Recall@0.4 | Precision | Dets/img |
|---|---|---:|---:|---:|
| COCO init | unsupervised (train dist.) | 0.000 | – | 0.0 |
| **Weak-pretrained (301)** | **unsupervised (train dist.)** | **0.217** | 0.516 | 57.3 |
| COCO init | supervised test | 0.000 | – | 0.0 |
| **Weak-pretrained (301)** | **supervised test** | **0.203** | 0.170 | 32.1 |

**That first pass under-measured the model, and the correction matters.** DeepForest's
inference default is `nms_thresh = 0.05`, which suppresses any box overlapping a
higher-scoring one by more than 5% IoU. The unsupervised pseudo-labels overlap heavily in
closed canopy (143 boxes/image on these tiles — see the overlays in section 7b), so the
model is structurally barred from reproducing them. Sweeping the postprocessing, on the
model's **own unsupervised training data**:

| `score_thresh` | `nms_thresh` | Recall@0.4 | Precision | Dets/img | GT/img |
|---:|---:|---:|---:|---:|---:|
| 0.1 (default) | 0.05 (default) | 0.212 | 0.524 | 57.9 | 143.4 |
| 0.05 | 0.4 | **0.415** | 0.213 | 279.4 | 143.4 |
| 0.01 | 0.4 | 0.415 | 0.204 | 291.6 | 143.4 |
| 0.01 | 0.6 | **0.489** | 0.117 | 600.0 (cap) | 143.4 |

So the network recalls roughly **0.42–0.49** of its own training labels once NMS stops
fighting it, not 0.22. **Do not quote the 0.217 figure without this caveat.**

The conclusion survives the correction, but with softer magnitude, and it splits the
difference between the two hypotheses the earlier notes offered:

- **It is not broken.** COCO init detects literally nothing — its heads are randomly
  initialized, so `assert_coco_trunk`'s 281 verified tensors never covered them. The
  pretrained network detects trees. Stage 1 learned something real.
- **It is not trained, either.** Recalling under half of its *own training labels* after
  20 epochs on 7.1M boxes is not convergence, and precision collapses from 0.52 to 0.12 as
  NMS relaxes — it is spraying low-confidence boxes rather than localizing confidently.
  Its maximum score across 48 images is **0.30**, consistent with the classification-head
  bias having moved only from −4.595 (the focal-loss prior) to −4.101.

So the whole-network ablation's premise — "the model *starts as a working tree detector*
and fine-tuning only has to adapt it" — is overstated. It starts as a weak, under-confident one.

## 4. Stage 2 then hits it with 100× the learning rate, with no warmup

| Stage | Optimizer | LR | Batch | Scheduler | Warmup |
|---|---|---:|---:|---|---:|
| 1 (pretrain) | SGD | **1e-4** | 8 | None (constant) | 0 |
| 2 (fine-tune) | SGD | **1e-2** | 32 | None (constant) | 0 |

`train.py` contains no scheduler code at all, and DeepForest's config defaults are
`scheduler.type: None`, `warmup_epochs: 0`. Whatever stage 1 produced is hit by gradients
two orders of magnitude larger than anything it saw, from the very first iteration of
stage 2. This is a sufficient mechanism for the null on its own, independent of stage-1
quality — and it predicts exactly the observed result: the whole-network arm landing on
*precisely* the COCO baseline (0.476 vs 0.476 WD).

## 5. Table 6's polygon transplant is architecturally confounded

Separate from the above, and it invalidates the polygon comparison specifically:

| | Mask R-CNN v2 (polygon model) | RetinaNet (box pretraining) |
|---|---|---|
| Backbone BN | `BatchNorm2d` (trainable) | `FrozenBatchNorm2d` |
| `backbone.body` tensors | 318 | 265 |
| Trainable backbone param tensors | 126 / 159 | 42 / 53 |

**0 of the 265 shared-name tensors are identical between MaskRCNN-v2-COCO and
RetinaNet-COCO** — they are different networks from different training recipes.

So the Table 6 arms are not "COCO backbone vs. weak-pretrained backbone":

- **Control**: Mask R-CNN v2 with its own COCO backbone, matched to the FPN/RPN/ROI heads
  it was trained with.
- **Treatment**: Mask R-CNN v2 with a *foreign RetinaNet* backbone bolted on — carrying
  FrozenBN running statistics that were never updated during pretraining — while the FPN
  and all heads stay tuned for the backbone that was just removed.

The transplant damage is confounded with the weak-supervision signal, and it points the
same way as the observed result (−0.050 AP40 OOD, 9 of 11 sources down).

**The missing control**: transplant a plain **RetinaNet-COCO** trunk (no weak pretraining)
into Mask R-CNN v2 and train it identically. That isolates transplant damage from
weak-label contribution. It has never been run, and without it Table 6 cannot attribute
its result to the unsupervised boxes.

## 6. What the data loader audit cleared

The loader was the other prime suspect. It is sound:

| Check | Result |
|---|---|
| Volume | 46,823 images, 7,126,326 boxes actually reach the model (5,853 steps × batch 8 = 46,824/epoch) |
| Diversity | **25 distinct NEON sites** (SOAP, BART, RMNP, WREF, TEAK, TALL, SERC, DELA, SJER, BLAN, JERC, CLBJ, YELL, BONA, HOPB, NOGP, OAES, CUPE, WLOU, REDB, ABBY, DEJU, OSBS, SRER, HEAL), years 2018–2019 |
| Degenerate boxes | 0 with `w<=0` or `h<=0`; 0 with a side > 400px |
| Box geometry | median 69×69px on 1024×1024 tiles; 152 boxes/image mean, max 530 |
| **Scale match** | after the `A.Resize(448,448)` the median unsupervised box is **30.2px** — *identical* to the domain-matched supervised source `Weecology_University_Florida` (400px tiles, median box 27px → **30.2px**) |
| Transfer code | `apply_box_pretrained_network` loads `strict=True` and refuses any partial cover; verified correct |
| Trunk actually trains | 42/53 backbone param tensors are trainable = **99.1% of trunk weight mass** (the 223 tensors bit-identical to COCO are 212 FrozenBN buffers + conv1/layer1) |

The one loader-adjacent caveat worth noting: 72,646 boxes (1.0%) have a side < 4px, which
after the 448 resize is under 2px — below the smallest RetinaNet anchor. That is a small
tail, not an explanation.

## 7. What is being run

`pretrain_backbone_for_polygons.py` now:

1. validates on **512 held-out SUPERVISED train images**, stratified by source, seeded
   (`build_supervised_val_subset()`) — supervised because the question is whether weak
   pretraining makes a better detector *on human labels*; loss on held-out pseudo-labels
   would only measure self-consistency. Taken from **train**, never test (contamination)
   and never the 59-image manuscript validation split (too narrow);
2. logs `val_loss` / `val_classification` / `val_bbox_regression` every epoch and dumps
   them into the `.out` log, so "did the loss drop?" is answerable without opening Comet;
3. selects and exports the **best** epoch by supervised `val_loss`, not the last;
4. supports `cosine` and `multistep-warmup` LR schedules (`--scheduler`), enabled by
   setting the `validation.csv_file` placeholder that DeepForest gates scheduling on;
5. scores the COCO init on the same holdout **before** training, so the results file
   reports a delta rather than an uninterpretable absolute.

**Smoke tests** `39879166` / `39879326` / `39879499` / `39879655` — three failures, three
real bugs (sections 7a and 7c), then a clean pass verifying all four acceptance criteria:
validation every epoch, `val_*` columns in `metrics.csv`, a real best-checkpoint path and
score, and a results file with BASELINE and AFTER blocks instead of `nan`.

**Job `39879762`** — `pretrain_backbone_supervised_val_lrsweep.sbatch`, array 0-2,
within-distribution, **running**:

| Task | Recipe | Purpose |
|---|---|---|
| 0 | batch 8, lr 1e-4, constant | the old recipe, now instrumented — the control |
| 1 | batch 32, lr 1e-3, cosine | moderate |
| 2 | batch 32, lr 1e-2, cosine | matches the stage-2 fine-tune LR, removing the 100× discontinuity |

**Decision rule.** If supervised `val_loss` falls substantially further in tasks 1–2 than
in task 0, stage 1 was undertrained, the null is an artifact, and the whole box and
polygon ablation ladder must be re-run from a converged backbone. If all three plateau at
the same place with 22%-recall detectors, the pretraining signal itself is the problem and
the next suspect is the quality of the Weinstein et al. 2018 pseudo-labels.

## 7a. A latent bug the fix exposed

Adding a validation loader immediately crashed the run twice, and the cause is worth
recording because it explains why nobody hit it before:

```
MisconfigurationException: Cannot use `LearningRateMonitor` callback with `Trainer` that has no logger.
```

`deepforest.main.create_trainer()` appends its own `LearningRateMonitor` whenever
validation is active, guarded on `if logger is not None`. This script passed
`logger=False` — and `False is not None` is `True`, so the callback was added to a
Trainer that Lightning then rejected for having no logger. The guard has always been
wrong; it was simply unreachable, because `has_val` was never true for a stage-1 run.
Fixed by passing `logger=None`, which makes DeepForest build a `CSVLogger` — so per-epoch
`val_loss` now also lands in `metrics.csv` inside the run's output directory.

## 7c. A third gate, found by the smoke test

Wiring in a validation loader was **not** sufficient. `create_trainer()` also passes
`check_val_every_n_epoch = config.validation.val_accuracy_interval`, and that default is
**20**. So even with the loader correctly attached, a 20-epoch run would have validated
exactly **once**, at the final epoch — `ModelCheckpoint` would have seen a single value,
"best" would have been identical to "last", and the entire instrumentation would have
silently reproduced the original bug.

The smoke test caught this: job `39879499` exited 0, wrote its exports, and confirmed the
cosine schedule was working (`lr-SGD` 1e-4 → 5.05e-5) — while `metrics.csv` contained no
`val_*` column at all and the checkpoint reported `best: <empty>` / `score: None`.

There are therefore **three independent gates** between "pass a val dataloader" and
"actually get a validation curve", all defaulting closed:

| # | Gate | Default | Effect when closed |
|---|---|---|---|
| 1 | `has_val` (empty test split) | closed | no ModelCheckpoint, no early stop, **no LR scheduler** |
| 2 | `logger is not None` with `logger=False` | mis-fires | `LearningRateMonitor` crashes the run |
| 3 | `check_val_every_n_epoch` = `val_accuracy_interval` | **20** | validation runs once at the last epoch, or never |

`--val-every-n-epochs` (default 1) now opens gate 3, setting both Lightning's
`check_val_every_n_epoch` (gates `validation_step`, i.e. `val_loss`) and
`config.validation.val_accuracy_interval` (gates `on_validation_epoch_end`, i.e.
`box_recall` / `box_precision` / mAP).

## 7b. Visual check of the unsupervised labels

Six tiles from six NEON sites (BART, SJER, TEAK, JERC, WREF, OSBS) were rendered with
their boxes overlaid. The pseudo-labels are **good**: boxes track crowns closely in
plantation rows (JERC), in sparse woodland (SJER, TEAK), and in closed canopy (BART 211
boxes, WREF 152); the near-empty grassland tile (OSBS) correctly carries 2 boxes.
Label quality is not a candidate explanation for the null.

One consequence visible in those tiles: in closed canopy the pseudo-labels **overlap each
other heavily**. DeepForest's inference default is `nms_thresh = 0.05`, which suppresses
any box overlapping a higher-scoring one by more than 5% IoU — so the model is
structurally unable to *reproduce* a dense label set at inference time, whatever it has
learned. NMS is inference-only and does not affect training loss, but it does cap the
zero-shot recall measured in section 3; see the threshold sweep there.

## 8. Consequences for what has already been written

- `notes/weak_supervision_pretraining_table.md` — sections 1, 2 and 4 report a null that
  is **not yet interpretable**. The ImageNet rung (initialization barely matters at full
  supervision) still stands on its own, since it does not depend on stage 1.
- `notes/weak_supervision_data_scaling.md` — the "dataset size is not the mechanism"
  conclusion is **suspended**. The grid tested transfer from an undertrained backbone.
- **Manuscript Table 6 must not be updated from the current numbers.** Both the
  undertrained stage 1 and the untested backbone transplant confound it.

---

## 9. Resolution: the loss-magnitude jump is the init change, and the LR sweep closes the undertraining hypothesis

*2026-08-22. Triggered by the observation that `train_loss` on older Comet runs sits far
below recent ones, casting doubt on every model that inherits stage 1.*

### 9a. The loss offset is `--init-mode`, present at epoch 0

Every stage-1 run before **2026-08-10** started from `weecology/deepforest-tree` — the
NEON-trained tree detector — because `pretrain_backbone_for_polygons.py` called
`model.load_model("weecology/deepforest-tree")` unconditionally. (`HEAD` still does, at
line 139; `--init-mode` arrives in `cc88174`, *"make sure we have the COCO baseline"*,
2026-08-10.) Every run since starts from COCO with **randomly initialized** RetinaNet
heads.

The gap is visible before a single gradient step:

| init | runs | `train_loss` epoch 0 | epoch 19 | relative drop |
|---|---|---:|---:|---:|
| `weecology/deepforest-tree` (pre-08-10) | 22 | 0.91 – 1.13 | 0.75 – 0.98 | −15% to −27% |
| `coco` (post-08-10) | 10 | **1.56 – 1.70** | 0.92 – 1.22 | **−36% to −41%** |

The two experiments in the challenge are one on each side of that line:

| Comet | date | init | e0 → e19 | drop |
|---|---|---|---|---:|
| `e9362a80` `boxes-out-of-distribution-pretrain-lr0.0001` | 08-05 | deepforest-tree | 0.962 → 0.766 | −20.4% |
| `32ab2c3b` `boxes-within-distribution-pretrain-supval-oldrecipe` | 08-21 | coco | 1.700 → 1.081 | −36.4% |

**The older curves are lower because the model already knew what a tree was.** That
inherited NEON supervision is exactly the contamination that made the pre-08-10 ablation
uninterpretable. Measured as *learning from the unsupervised boxes*, the recent runs do
strictly more work. The absolute levels are not comparable across 08-10 and never were.

### 9b. The LR sweep resolves the decision rule in section 7 — against undertraining

Job `39879762`, within-distribution, all `init-mode coco`, first stage-1 runs ever to
carry a validation set:

| Task | Recipe | `train_loss` e0→e19 | `val_loss` e0→e19 | `val_bbox_regression` |
|---|---|---|---|---|
| 0 | bs 8, lr 1e-4, constant | 1.70 → **1.08** (−36%) | 1.79 → 1.75 (noisy, min 1.43 @e18) | 0.70 → 0.69 **flat** |
| 1 | bs 32, lr 1e-3, cosine | 1.56 → **0.92** (−41%) | 1.42 → **3.42** (diverges) | 0.69 → 0.67 **flat** |
| 2 | bs 32, lr 1e-2, cosine | NaN by e3 | NaN | NaN |

The rule was: *if supervised `val_loss` falls substantially further in tasks 1–2, stage 1
was undertrained and the null is an artifact.* **It does the opposite.** Task 1 fits the
pseudo-labels best of any run on record and its supervised `val_loss` gets **2.4× worse**,
driven entirely by `val_classification` (0.73 → 2.76).

The sharpest number is the one that never moves: **`val_bbox_regression` sits at ~0.69 in
every arm, at every learning rate, across all 20 epochs.** Localization — the single thing
the ablation hoped would transfer — does not improve on human labels under any recipe.

Stage-1 AP40 on the 512 held-out supervised images, against its own pre-training baseline:

| Arm | BASELINE (COCO init) | AFTER pretraining |
|---|---:|---:|
| oldrecipe lr1e-4 | 0.000 | **0.006** |
| lr1e-3 cosine | 0.000 | **0.001** |

(At DeepForest inference defaults, `nms_thresh=0.05` — which section 3 shows suppresses
heavily on dense canopy. Both columns use the identical eval, so the *delta* is sound.)

**So the null is no longer suspended for the undertraining reason.** Stage 1 is weak, but
"train it harder" has now been tested and makes supervised transfer worse, not better.
The finding is about the Weinstein et al. 2018 pseudo-labels — the model learns to
reproduce their confidence pattern without learning better tree extent — not about the
recipe. The `lr1e-2` NaN arm is a separate defect, addressed by the new `--grad-clip 1.0`
default; re-run it before calling the sweep complete.

### 9c. Two traps to avoid when acting on this

**`--ckpt-monitor val_loss` selected epoch 0 on the lr1e-3 arm.** Its best score is
1.4218 — the epoch-0 value. Because `val_loss` only ever rises, "best checkpoint" is the
*least trained* model, and `pretrain_supval_lrsweep/lr1e-3-cosine/box_backbone_*.pt` is
therefore close to a plain COCO trunk. Any downstream run fed that export would produce a
null for trivial reasons. Task 0 is fine (score 1.4297 = epoch 18). Monitor
`box_recall`/AP40 rather than `val_loss` if these exports are to be reused.

**Which downstream arms inherit contaminated weights** (`outputs/pretrain/` = pre-08-10 =
deepforest-tree init):

| Downstream sbatch | consumes | status |
|---|---|---|
| `train_polygons_box_pretrained.sbatch` (**Table 6**) | `outputs/pretrain/` | **contaminated** |
| `train_polygons_box_pretrained_annotationsafecrop.sbatch` | `outputs/pretrain/` | **contaminated** |
| `train_boxes_box_pretrained{,_coco}.sbatch` | `outputs/pretrain/` | **contaminated** |
| `train_boxes_box_pretrained_coco_v023.sbatch` | `outputs/pretrain_v023/` | **contaminated** |
| `train_polygons_box_pretrained_v023_cleaninit.sbatch` | `outputs/pretrain_v023_cocoinit/` | clean init |
| `train_boxes_box_pretrained_coco_v023_cleaninit.sbatch` | `outputs/pretrain_v023_cocoinit/` | clean init |
| `train_boxes_box_pretrained_full_v023.sbatch` | `outputs/pretrain_v023_cocoinit_full/` | clean init |

The clean-init rows are uncontaminated by tree labels but still inherit an
**undertrained** stage 1 — which section 9b now shows is not fixable by training longer.
The remaining live confound for polygons is unchanged: the architectural transplant of
section 5, whose control (plain RetinaNet-COCO trunk into Mask R-CNN v2) **still has not
been run** and is the one experiment Table 6 cannot be written without.
