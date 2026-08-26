# Boxes weak-supervision ablation, out-of-distribution split (v0.23) — why the loss curves mislead

Written 2026-08-25. Prompted by a Comet compare of
`boxes-out-of-distribution-coco-lr0.01-cotrain-weak` (d4d843be) vs
`boxes-out-of-distribution-box_pretrained_full-lr0.01-converged` (79376ad7) on a **step**
x-axis, where the pretrained arm appears to converge far faster while the final numbers are
flat. The apparent speed-up is a plotting artifact; the flat final numbers are real.

## The three arms

All three: RetinaNet, SGD lr **0.01**, no scheduler (constant lr, confirmed by `lr-SGD`),
momentum 0.9, batch 32, grad clip 1.0, early stop patience 10 on `box_recall`,
`ModelCheckpoint(monitor="box_recall", mode="max")`, split `out-of-distribution`, v0.23.

| Comet | arm | init | train data | images | steps/epoch |
|---|---|---|---|---|---|
| 2305d48a `…coco-lr0.01-converged-ctl` | CTL | torchvision COCO | `TreeBoxes_supervised_v0.23` | 12,752 | 398 |
| 79376ad7 `…box_pretrained_full-lr0.01-converged` | pretrain→finetune | weak-pretrained net (`pretrain_converged/lr1e-3-cosine-clip`) | `TreeBoxes_supervised_v0.23` | 12,752 | 398 |
| d4d843be `…coco-lr0.01-cotrain-weak` | co-train | torchvision COCO | `TreeBoxes_v0.23`, `--include-unsupervised --exclude-sources '*Young*'` | 59,575 | 1,861 |

**There is no learning-rate difference.** The only parameter diffs between the two compared
runs are `init-mode`, `box-backbone-checkpoint`, `include-unsupervised`, `exclude-sources`,
`root_dir`, and `output-dir`.

## Artifact 1: the step axis is not a common clock

The co-train arm sees 4.68× more images per epoch (1,861 vs 398 steps). At step 9,575 the
pretrained arm has run 23 epochs; the co-train arm has run ~5. Anything plotted against
step compresses the pretrained arm's curve into the left edge and makes it look steeper.
Per **epoch** the two val-loss curves sit on top of each other (ep4: 0.761 vs 0.770; ep6:
0.750 vs 0.739; ep10: 0.717 vs 0.749).

## Artifact 2: the training losses are over different datasets

Co-train's `train_loss_epoch` is computed over 6.4M pseudo-boxes (Weinstein et al. 2018
unsupervised) mixed with the human labels, so it has an irreducible label-noise floor and
plateaus at 0.664. The pretrained arm's is over 12,752 clean images and falls to 0.500.
These are not the same objective and cannot be compared. The comparison that *is* valid —
pretrained vs CTL, identical train set — runs the other way: at epoch 16 the pretrained arm
is at **0.586** train loss, CTL at **0.545**. Weak pretraining does not speed up fitting.

## Artifact 3: the compared val losses diverge because one arm overfits

Val loss is comparable (both eval on the same 3,809-image test split — `exclude_sources`
applies to all splits, but both weak sources are train-only, and the two packaged releases
carry byte-identical test splits: 316,576 boxes / 3,809 images / same 4 sources). The
pretrained arm bottoms at **0.712 @ ep15** then climbs to 0.789 by ep23; the co-train arm
holds flat at 0.74–0.76. The "faster drop" ends in overfitting, not a better model. Reported
metrics come from the best-`box_recall` checkpoint, so this does not corrupt the score.

## What the clean per-step comparison (pretrained vs CTL, same clock) shows

| epoch | CTL val_loss / recall / map_50 | pretrained val_loss / recall / map_50 |
|---|---|---|
| 0 | 0.970 / 0.414 / 0.171 | 1.097 / 0.355 / 0.101 |
| 2 | 0.787 / 0.552 / 0.316 | 0.829 / 0.505 / 0.295 |
| 4 | 0.750 / 0.564 / 0.343 | 0.761 / 0.543 / 0.328 |
| 6 | 0.725 / **0.596** / 0.370 | 0.750 / 0.556 / 0.350 |
| 13 | 0.725 / 0.588 / **0.376** | 0.715 / **0.584** / **0.372** |
| best | 0.712 / 0.596 / 0.376 | 0.712 / 0.584 / 0.372 |

The weak-pretrained network starts **behind** COCO and takes ~10 epochs to catch up; it
never passes it. Time-to-`val_loss<0.75`: CTL step 1,994 · pretrained step 2,792 · co-train
step 13,033. On every clock, plain COCO is the fastest arm.

## Final test-split scores (best checkpoint, full MillionTrees eval)

| arm | AP40 | AP60 | recall | mask-aware precision | accuracy | worst-group AP40 |
|---|---|---|---|---|---|---|
| CTL (COCO, supervised only) | **0.519** | 0.338 | **0.668** | **0.784** | **0.402** | 0.299 |
| pretrain→finetune (weak) | 0.513 | **0.344** | 0.655 | 0.779 | 0.396 | 0.305 |
| co-train (weak) | 0.512 | 0.333 | 0.655 | 0.771 | 0.389 | **0.308** |

## Conclusion

On the OOD boxes split at v0.23, **neither route to the weak labels beats the plain COCO
baseline** — the spread (AP40 0.512–0.519, recall 0.655–0.668) is within run-to-run noise
and the sign favors the control. The visual impression that weak pretraining "learns much
faster" comes entirely from comparing it against the co-training arm on a step axis with
4.68× different steps-per-epoch. There is no hidden win for the final metrics to fail to
reflect.

Caveats: n=1 per arm, no seed sweep, and the two weak arms differ in more than one factor
from each other (co-train also excludes Young et al. 2025, which the pretrained backbone
never saw either — that part is controlled). See
[[project_deepforest_tree_default_contaminates_ablations]] for why `model.name=None` matters
here; these runs are on the clean path.
