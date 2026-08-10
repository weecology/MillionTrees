# Box task: training on all sources vs. exhaustively-annotated sources only

> **AP columns are AP50** (IoU 0.5), the metric in force when these runs were made. AP is now
> AP40 (IoU 0.4) project-wide, so these AP values are historical and are not comparable to the
> current leaderboard.

Does restricting the training set to **exhaustively-annotated ("complete")
sources** improve the box detector, or does the extra (incompletely-annotated)
data help despite its label noise?

We train DeepForest (RetinaNet) on TreeBoxes v0.19 under two training regimes
and evaluate **both on the same, unchanged test set**:

- **All sources** — every supervised box source in the train split (the
  standard leaderboard training set).
- **Complete only** — train restricted to sources flagged
  `complete=True` in `data_prep/source_completeness.csv` (exhaustively
  annotated). The filter is applied to the **train split only**
  (`--remove-incomplete`); the validation/test sets are byte-for-byte identical
  to the "all sources" run, so the two regimes are directly comparable.

Unsupervised sources are excluded from both regimes (the leaderboard default).

| Split | Train regime | Train images | Test images |
|---|---|---:|---:|
| within-distribution | All sources | 10,460 | 2,670 |
| within-distribution | Complete only | 2,354 | 2,670 |
| out-of-distribution | All sources | 9,173 | 3,957 |
| out-of-distribution | Complete only | 1,583 | 3,957 |

## Within-distribution

| Train regime | LR | Accuracy | Recall | Mask-aware precision | AP50 |
|---|---:|---:|---:|---:|---:|
| All sources | 0.005 | **0.317** | **0.546** | 0.783 | **0.310** |
| Complete only | 0.005 | 0.286 | 0.458 | 0.741 | 0.231 |
| **Δ (complete − all)** | | **−0.031** | **−0.088** | **−0.042** | **−0.079** |
| All sources | 0.001 | 0.314 | 0.539 | 0.797 | 0.290 |
| Complete only | 0.001 | 0.285 | 0.422 | 0.786 | 0.214 |
| **Δ (complete − all)** | | **−0.029** | **−0.117** | **−0.011** | **−0.076** |

## Out-of-distribution

| Train regime | LR | Accuracy | Recall | Mask-aware precision | AP50 |
|---|---:|---:|---:|---:|---:|
| All sources | 0.005 | **0.349** | **0.517** | **0.891** | **0.290** |
| Complete only | 0.005 | 0.305 | 0.423 | 0.879 | 0.148 |
| **Δ (complete − all)** | | **−0.044** | **−0.094** | **−0.012** | **−0.142** |
| All sources | 0.001 | 0.315 | 0.506 | 0.861 | 0.269 |
| Complete only | 0.001 | 0.325 | 0.435 | 0.889 | 0.180 |
| **Δ (complete − all)** | | **+0.010** | **−0.071** | **+0.028** | **−0.089** |

## Summary

Restricting training to exhaustively-annotated sources **reduced** detector
performance in every configuration: recall fell by 0.07–0.12 and AP50 by
0.08–0.14 across both splits and both learning rates. The drop is consistent
despite the test set being identical, so it reflects the loss of training signal
from the incomplete sources (≈4–6× fewer training images) rather than an easier
or harder evaluation. Mask-aware precision was roughly unchanged (the model that
sees more data is not simply over-predicting). The lone exception is a small
accuracy gain on the out-of-distribution split at lr 0.001 (+0.010), within
run-to-run noise. **Including incompletely-annotated sources in training
improves the box detector**, so they are retained in the standard training set.

---

*Source data (TreeBoxes v0.19, identical test sets per split):*

- *All sources:* `training/boxes/outputs/<split>_lr<lr>/results_<split>.txt`
  (lr-sweep job 35574489)
- *Complete only:* `training/boxes/outputs/<split>_lr<lr>_completeonly/results_<split>.txt`
  (paired job 35683347, `--remove-incomplete`)

*Metrics are the MillionTrees "Average" (macro across source groups) at the
default `eval_score_threshold=0.1`. Last updated 2026-06-24.*
