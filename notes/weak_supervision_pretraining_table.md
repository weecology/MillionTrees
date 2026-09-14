# Weak-supervision box pretraining ablations (v0.23 / AP40)

> ## ⚠️ CONCLUSIONS SUSPENDED — 2026-08-21
>
> An audit of the pretraining stage (`notes/weak_supervision_stage1_audit.md`) found that
> **stage 1 was never validated, never model-selected, and never LR-scheduled**, and is
> badly undertrained: the exported network recalls only **22% of its own training labels**
> zero-shot. Stage 2 then fine-tunes at **100× the stage-1 learning rate with no warmup**.
>
> The nulls in sections 1, 2 and 4 below are therefore **not yet interpretable as evidence
> that the unsupervised boxes add nothing** — they may be measuring an undertrained
> backbone. Table 6 (section 2) carries a second, independent confound: the polygon
> transplant grafts a foreign RetinaNet backbone into Mask R-CNN v2, whose own COCO
> backbone shares **0 of 265 tensors** with it.
>
> **Do not update manuscript Table 6 from these numbers.** The ImageNet rung in section 1
> is unaffected and still stands. Corrected runs are in flight — see the audit note.



*Last updated 2026-08-21. **All arms of both ablations are now complete.** Supersedes the
2026-05-18 v0.13/mAP version of this file, whose numbers came from a contaminated comparison —
see "Why the older numbers are gone" below.*

The question in both ablations: **does inserting a pretraining stage on the 6.4M unsupervised
boxes (Weinstein et al. 2018, 41,691 images) beat training straight from torchvision COCO?**

**Answer: no, in either geometry.** Box to box is a null at both transfer depths and both splits.
Box to polygon is a null within-distribution and a *degradation* out-of-distribution. The
previously reported +0.083 AP40 polygon gain does not survive a corrected control.

The result that explains the nulls is the new ImageNet rung: **removing detection pretraining
entirely costs only 0.020 AP40 within-distribution and 0.006 out-of-distribution.** At MillionTrees
scale the initialization is nearly irrelevant to the box task, so there was never headroom for
weak-label pretraining to occupy.

Every arm here starts from a **verified** torchvision init (`assert_coco_trunk()` /
`assert_imagenet_trunk()` compare every backbone/FPN tensor against a freshly built torchvision
model and raise on mismatch), so any difference is attributable to the pretraining stage and
nothing else. All banners were confirmed present in the job logs.

---

## 1. Box to box — COMPLETE (four-rung initialization ladder)

Jobs `39129719` (pretrain trunk) / `39129720` (COCO control) / `39129721` (trunk-only treatment);
`39193929_0` + `39614225` (whole-network pretrain) / `39614224` + `39614226` (whole-network
treatment); `39708673` (ImageNet cold-start rung) — all COMPLETED 0:0.

Four initializations, one recipe (batch 32, lr 0.01, grad-clip 1.0, patience 10/15):

- **ImageNet**: ResNet50 IMAGENET1K_V1 trunk, freshly initialized FPN and heads. *No detection
  pretraining of any kind.* The cold-start floor.
- **COCO** (control): torchvision RetinaNet COCO_V1, all 281 backbone/FPN tensors verified.
- **COCO + weak trunk-only**: `backbone.body.*`, 265 of 301 tensors. FPN reset to COCO, both
  detection heads reset to random.
- **COCO + weak whole network**: all 301 tensors, `strict=True`. The model *starts as a working
  tree detector* and fine-tuning only has to adapt it.

### Within-distribution

| Init | Recall | Mask-aware precision | AP40 | Counting MAE |
|---|---:|---:|---:|---:|
| ImageNet (no detection pretraining) | 0.632 | 0.584 | 0.456 | 15.95 |
| COCO only (control) | 0.637 | 0.633 | **0.476** | 14.89 |
| COCO + weak trunk (265/301 tensors) | 0.635 | 0.642 | 0.460 | 14.58 |
| COCO + weak whole network (301/301) | 0.631 | 0.644 | **0.476** | 14.54 |
| **Δ, ImageNet − COCO** | −0.005 | −0.049 | **−0.020** | +1.06 |
| **Δ, weak trunk − COCO** | −0.002 | +0.009 | **−0.016** | −0.31 |
| **Δ, weak whole network − COCO** | −0.006 | +0.011 | **0.000** | −0.35 |

### Out-of-distribution

| Init | Recall | Mask-aware precision | AP40 | Counting MAE |
|---|---:|---:|---:|---:|
| ImageNet (no detection pretraining) | 0.660 | 0.778 | 0.511 | 15.38 |
| COCO only (control) | 0.668 | 0.777 | **0.517** | 15.10 |
| COCO + weak trunk (265/301 tensors) | 0.646 | 0.771 | 0.502 | 16.21 |
| COCO + weak whole network (301/301) | 0.642 | 0.783 | 0.501 | 15.16 |
| **Δ, ImageNet − COCO** | −0.008 | +0.001 | **−0.006** | +0.28 |
| **Δ, weak trunk − COCO** | −0.022 | −0.006 | **−0.015** | +1.11 |
| **Δ, weak whole network − COCO** | −0.026 | +0.006 | **−0.016** | +0.06 |

**The whole span of the ladder is 0.020 AP40 within-distribution and 0.016 out-of-distribution.**
Those four initializations differ about as much as two initializations can — one has never seen a
detection task, one is a fully trained tree detector — and they finish within noise of each other.

**Result: null at both transfer depths.** Pretraining on 6.4M unsupervised boxes does not improve
the box task on either split. Trunk-only moves AP40 −0.016 / −0.015; the whole network moves it
0.000 / −0.016.

**The whole-network row closed the truncation loophole.** The trunk-only transfer discarded the FPN
(16 tensors) and both detection heads (20 tensors) — every layer that encodes *"this is a tree and
this is its extent"* — because `export_backbone_weights()` filtered on the `"backbone.body."`
prefix, a filter inherited from the box-to-polygon ablation. That was a live explanation for the
null. It is no longer available: handing fine-tuning a complete, verified 301-tensor tree detector
lands on **exactly** the COCO baseline within-distribution (0.476 vs 0.476) and slightly below it
OOD (0.501 vs 0.517).

**The ImageNet row explains why.** The natural reading of a null had been *volume* — that 13,538
images / 796k boxes reaches the same solution from any starting point. That is now measured rather
than assumed, and it holds in the strongest possible form: an initialization with **no detection
pretraining at all** reaches 0.456 / 0.511 AP40, within 0.020 / 0.006 of COCO. Fine-tuning on the
full supervised split does not merely erase the weak-label pretraining, it erases essentially
every initialization difference. There is no room above the COCO baseline for weak labels to add.

The follow-up test of the volume hypothesis — shrinking supervision 65× — also came back null;
see `notes/weak_supervision_data_scaling.md` (job `39708566`) and section 4 below.

*(Caveat carried from the submit note: ImageNet and COCO trunks already agree on 223/265 tensors,
since COCO RetinaNet is itself ImageNet-initialized. The ImageNet−COCO delta therefore rides on 42
trunk tensors plus the 16-tensor FPN and 20-tensor heads, and a small gap is expected. It is still
the largest gap in the ladder.)*

## 2. Box to polygon (manuscript Table 6) — COMPLETE

Jobs `39185675_0` (WD control), `39626339` (OOD control, eval-only recovery of the `39185675_1`
epoch-19 checkpoint), `39614228` (both box-pretrained arms) — all COMPLETED 0:0. Mask R-CNN v2,
366 backbone/FPN tensors verified against COCO in both arms; the treatment swaps 265
`backbone.body.*` tensors for the unsupervised box trunk, leaving 98 at COCO init.

**This is the first version of Table 6 where neither arm has seen a NEON label and both arms start
from a genuinely pretrained backbone.**

| Split | Init | Recall | Mask-aware precision | AP40 |
|---|---|---:|---:|---:|
| within-distribution | COCO only (control) | 0.558 | 0.759 | **0.191** |
| within-distribution | Box-pretrained backbone | 0.549 | 0.781 | 0.183 |
| out-of-distribution | COCO only (control) | 0.579 | 0.581 | **0.189** |
| out-of-distribution | Box-pretrained backbone | 0.547 | 0.550 | 0.139 |
| **Δ within-distribution** | | −0.009 | +0.022 | **−0.008** |
| **Δ out-of-distribution** | | −0.032 | −0.031 | **−0.050** |

**Within-distribution is a null, and a fragile one.** The −0.008 macro delta matches the
image-weighted delta (0.229 → 0.221), but it is not stable at the source level: a single 18-image
source (Safonova et al. 2021) collapses 0.960 → 0.051 and contributes −0.036 of the macro delta on
its own; excluding it the macro delta is +0.028. Read this row as "no measurable difference,"
not as a small loss.

**Out-of-distribution is a real degradation.** Unlike the WD row it is consistent across sources
rather than driven by outliers: **9 of 11 test sources drop**, including the four largest —
Lefebvre (n=1322) −0.108, Cloutier (n=425) −0.042, Troles (n=313) −0.027, SelvaMask (n=264)
−0.008. Image-weighted AP40 falls 0.263 → 0.199. Substituting a NEON-temperate weak-label trunk
for the COCO trunk measurably hurts generalization to unseen polygon domains.

**The old +0.083 AP40 gain is gone.** It measured *pretrained vs. randomly initialized*, because
`deepforest_polygon.yaml` routed into `MaskRCNN.from_pretrained()` with a RetinaNet checkpoint
whose keys never matched (see bug 2 below). The manuscript Table 6 claim must be reversed: from
"unsupervised pretraining improves the polygon task" to **"unsupervised box pretraining gives no
benefit within-distribution and degrades out-of-distribution performance."**

## 3. Outstanding runs

| # | Run | Status | GPU cost |
|---|---|---|---:|
| 1 | Box-to-box whole-network pretrain + train, both splits | **DONE** — `39614224` / `39614225` / `39614226` | — |
| 2 | Polygon OOD control **eval only** | **DONE** — `39626339` | — |
| 3 | Polygon box-pretrained, both splits | **DONE** — `39614228` | — |
| 4 | Box data-scaling grid (3 train sizes × 3 inits) | **DONE** — `39708566`, see section 4 | — |
| 5 | ImageNet cold-start rung, both splits | **DONE** — `39708673` | — |
| 6 | **Zero-shot eval of the stage-1 network on the box test split** | **not run — the one remaining gap** | ~0.5 h |
| 7 | ImageNet arm added to the data-scaling grid | not run — optional | ~3 × 1.5 h |

**Run 6 is the only thing that would still change the interpretation.** Stage-1 quality has never
been measured: `training/weak_supervision/outputs/pretrain_*/<split>/results_<split>.txt` are all
`nan`, because that stage evaluates on the unlabeled pretraining split. Scoring
`box_network_<split>.pt` (301 tensors, 129 MB, verified on disk) zero-shot against the standard box
test split separates the two remaining stories cleanly:

- Zero-shot AP40 is respectable (~0.3+) → the weak-label features are real and genuinely redundant
  with what supervision provides. The null is a headroom result and is publishable as one.
- Zero-shot AP40 is near zero → the pretraining stage never produced a tree detector, and every
  null in this file is measuring a broken stage 1 rather than a saturated benchmark.

Lesson kept from the failures: do not chain a whole array behind one `afterok`. The original
`submit_box_pretraining_ablation_full.sh` did, so one task's quota death cancelled both splits;
`submit_weak_supervision_recovery.sh` submits them separately.

## 4. Does supervision volume explain the null? — No

Job `39708566`, full detail in `notes/weak_supervision_data_scaling.md`. Shrinking the supervised
train split from 13,538 images to 206 (65×) while holding the test set fixed at 3,023 images does
**not** open a gap for the pretrained init. Image-weighted ΔAP40 (whole network − COCO) by tier:
−0.011 at 206 images, −0.036 at 914, −0.005 at 2,056, +0.003 at 13,538. The positive *macro*
deltas at two tiers are macro-averaging artifacts — a single n=1 source supplies most of them.

The one thing supervision volume clearly does control is **damage**: the trunk-only init, which
resets the FPN and both heads, costs −0.121 AP40 at 206 images and converges to −0.016 at full
data. Large supervision erases the harm of a bad initialization; it never converts the weak-label
initialization into a gain.

## Why the older numbers are gone

Three successive generations of this experiment were invalidated by initialization bugs, each
found only after the run:

1. **Box side, deepforest-tree contamination.** `deepforest`'s config default is
   `model.name="weecology/deepforest-tree"` and `deepforest.__init__` loads it before `train.py`
   runs, so `--init-mode coco` never gave torchvision COCO — it gave the NEON-trained detector.
   Both arms were already tree detectors differing by a ~6% nudge. Fixed by forcing
   `model.name=None` plus `assert_coco_trunk()`.
2. **Polygon side, random initialization.** `deepforest_polygon.yaml` sets no `model.name`, so the
   base default applied and `create_model()` routed into `MaskRCNN.from_pretrained()`, which builds
   a cold-start shell and then loads a *RetinaNet* checkpoint whose keys never match a Mask R-CNN.
   Nothing reached the backbone: 0/318 `backbone.body` tensors equalled COCO_V1 and all 53 BN
   `running_mean` were exactly 0. The old **+0.083 AP40** Table 6 gain therefore measured
   *pretrained vs. random*, which inflates the benefit — the opposite direction from bug 1.
   **Corrected in section 2: the gain becomes −0.008 WD / −0.050 OOD.**
3. **Truncated transfer.** Section 1's old caveat. **Closed** by the 301-tensor whole-network arm.

The v0.13 numbers previously in this file (random split, +0.305 recall / +0.031 mAP) came from
generation 2 and are not comparable to anything here. They are also AP50-era; see CLAUDE.md item 4.

---

*Source data:*
- *Box ImageNet:* `training/boxes/outputs/imagenet_v023/<split>/results_<split>.txt`
- *Box COCO control:* `training/boxes/outputs/coco_v023_cleaninit/<split>/results_<split>.txt`
- *Box trunk-only:* `training/boxes/outputs/box_pretrained_coco_v023_cleaninit/<split>/results_<split>.txt`
- *Box whole-network:* `training/boxes/outputs/box_pretrained_full_v023/<split>/results_<split>.txt`
- *Polygon COCO control:* `training/weak_supervision/outputs/polygon_coco_v023_cleaninit/<split>/results_<split>.txt`
- *Polygon box-pretrained:* `training/weak_supervision/outputs/polygon_box_pretrained_v023_cleaninit/<split>/results_<split>.txt`
- *Data scaling:* `training/boxes/outputs/datascale_v023/<tier>/<init>/results_within-distribution.txt`
