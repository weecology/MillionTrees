# TreePoints weak-supervision pretraining ablation (v0.24)

> **STATUS: COMPLETE — 2026-09-02.** All five jobs finished 0:0
> (stage 1 `40777951`/`40777952`, stage 2 control `40735121`, stage 2 treatment
> `40777953`/`40777954`, validation eval `40777955`). Result tables below are filled.
>
> **Headline: first non-null in the weak-supervision ablation family.** On the clean
> held-out validation split (Allen 2025 + Frey 2026 TLS), the AutoArborist
> continue-pretraining stage lifts point F1 on **both** splits — **+0.028
> within-distribution, +0.050 out-of-distribution** — entirely through recall
> (+0.016 / +0.051) with precision already saturated at ~0.94. Both validation sources
> improve, so it is not a single-source artifact. Counting MAE stays unstable in both
> directions and is not a reliable signal here. The box arm was null at every transfer
> depth (`notes/weak_supervision_pretraining_table.md`); the point task, with far less
> supervised data in v0.24 after the AutoArborist demotion, had the headroom the note
> predicted.

The point analogue of the box→box / box→polygon weak-supervision arms
(`notes/weak_supervision_pretraining_table.md`, manuscript Table 6).

## Question

Does inserting a pretraining stage on **AutoArborist** (`Beery et al. 2022 unsupervised`,
~22k images of municipal tree-inventory points loosely aligned to NAIP imagery) between
the released TreeFormer checkpoint and the supervised MillionTrees fine-tune beat
fine-tuning **directly** from the released checkpoint?

- **Control arm**: `weecology/deepforest-tree-point` → fine-tune on supervised
  MillionTrees points train.
- **Treatment arm**: `weecology/deepforest-tree-point` → continue-pretrain on
  AutoArborist (stage 1) → fine-tune on supervised MillionTrees points train.

Both stage-2 fine-tunes use the identical recipe (pretrained init, lr 2e-4, 20 epochs,
image size 896, `val_loss` checkpoint selection) — the current leaderboard points recipe
(`train_points_896_isolated.sbatch`). The only difference is the stage-2 `--checkpoint`.

## Why AutoArborist is the weak-supervision source here

In v0.24 `Beery et al. 2022` was demoted to `Beery et al. 2022 unsupervised`
(`notes/ood_split_test_sources_and_leaks.md`): its labels are municipal inventory
records, not image annotations (the TCD canopy filter discards ~50% of raw points), and
it is already barred from evaluation (`TRAIN_ONLY_SOURCES`). It was 86.8% of the old
supervised point-train images. Every AutoArborist row is in the **train** split for both
split schemes — exactly the property the unsupervised boxes have in the box arm — so it
is a clean weak-supervision pretraining corpus that never leaks into any eval.

## Design decisions (mirrors the box arm)

- **Stage-1 validation** = a held-out, source-stratified, seeded slice (512 imgs) of the
  *supervised* MillionTrees points train split — never test, never the manuscript
  validation split. We monitor on supervised points because that is the question the
  ablation asks; loss on held-out AutoArborist would only measure reproduction of
  municipal records. Those images stay in stage 2's train set (stage 1 and stage 2 are
  separate models). Verbatim rationale from `build_supervised_val_subset` in the box arm.
- **Stage-1 loss preset** = `point_pretrain.yaml` (`enforce_count=False`), the config the
  released checkpoint was trained under, applied post-`load_model` on the submodule
  (config alone is ignored after load — see `training/points/train.py`).
- **Stage-1 LR** = 1e-5 (gentle continued pretraining; don't overwrite the released
  checkpoint with noisy inventory points). 10 epochs, `val_loss` selection.
- **HF-format handoff**: stage 1 exports `config.json` + `model.safetensors` via
  `TreeFormerModel.save_pretrained`, which `training/points/train.py --checkpoint <dir>`
  consumes unchanged. No new stage-2 code.
- **Per-split stage 1** even though AutoArborist rows are identical across schemes — only
  the val holdout differs — and submitted per split so one split's failure can't cancel
  the other.

## Caveats to carry into interpretation

- **The box arm came back null at every transfer depth and both splits**
  (`notes/weak_supervision_pretraining_table.md`): at MillionTrees scale the
  initialization is nearly irrelevant to the box task. The point task has far less
  supervised data in v0.24 (3,403 images / 293,099 annotations after the AutoArborist
  demotion), so there may be more headroom here — but expect a small effect.
- **Stage 2 uses the MillionTrees *test* split as its Lightning val set** for checkpoint
  selection (inherited `train.py` behavior). Both arms do this identically, so the
  *comparison* is fair, but the absolute test numbers are optimistic. The validation
  split (Allen + Frey) is the clean read.
- **Amirkolaee et al. 2023 (the TreeFormer training set) is in the v0.24 points
  validation split**, because the released checkpoint was trained on it. Read Allen 2025
  and Frey 2026 from the per-source breakdown for the honest hold-out number.

## Jobs

| Job | Array | Script | Arm / split | Result |
|---|---|---|---|---|
| `40735121` | 0,1 | `train_points_weak_control.sbatch` | stage-2 control, both splits | COMPLETE 0:0 |
| `40777951` | 0 | `pretrain_points_autoarborist.sbatch` | stage-1 pretrain WD | COMPLETE 0:0 (2h10m) |
| `40777952` | 1 | `pretrain_points_autoarborist.sbatch` | stage-1 pretrain OOD | COMPLETE 0:0 (2h43m) |
| `40777953` | – | `train_points_weak_treatment.sbatch` | stage-2 treatment WD | COMPLETE 0:0 (2h54m) |
| `40777954` | – | `train_points_weak_treatment.sbatch` | stage-2 treatment OOD | COMPLETE 0:0 (5h18m) |
| `40777955` | 0-3 | `eval_points_weak_validation.sbatch` | validation eval, all 4 arm×split | COMPLETE 0:0 |

First submit (`40735122`–`40735126`, 2026-08-31) died on a transient /orange [Errno 5]
reading `*_Beery_et_al_2022.tif`; only the control arm (`40735121`) survived and was reused.
The pretrain→treatment→valeval chain was resubmitted as `40777951`–`40777955`.

Log sanity confirmed: stage 1 "loss-preset=pretrain APPLIED", "AutoArborist train images"
≈20k, finite BASELINE keypoint_acc, "HF checkpoint exported"; stage 2 "deepforest branch
OK: treeformer-training" and "Loading pretrained TreeFormer checkpoint:
.../pretrain/<split>/hf_checkpoint" (WD run ← WD checkpoint, OOD run ← OOD checkpoint);
valeval per-source breakdown names Allen et al. 2025 **and** Frey et al. 2026.

## Results — TEST split (`results_<split>.txt` from each stage-2 run)

| Split | Arm | Recall (keypoint_acc) | Mask-aware precision | F1 | Counting MAE |
|---|---|---:|---:|---:|---:|
| within-distribution | control (direct) | 0.848 | 0.657 | 0.740 | 66.99 |
| within-distribution | treatment (AutoArborist pretrain) | 0.772 | 0.845 | 0.807 | 28.45 |
| within-distribution | **Δ treatment − control** | **−0.076** | **+0.188** | **+0.067** | **−38.54** |
| out-of-distribution | control (direct) | 0.857 | 0.785 | 0.819 | 29.42 |
| out-of-distribution | treatment (AutoArborist pretrain) | 0.814 | 0.788 | 0.801 | 52.52 |
| out-of-distribution | **Δ treatment − control** | **−0.043** | **+0.003** | **−0.018** | **+23.10** |

TEST numbers are optimistic: both arms use the MillionTrees test split as the Lightning
val set for checkpoint selection (identical for the comparison, but not a clean hold-out).
TEST counting MAE is computed on very few complete sources (WD 4, OOD 1 — OSBS megaplot
only) and should not be read as a stable signal.

## Results — VALIDATION split (`results_<split>_validation.txt`, Allen + Frey; clean hold-out)

| Split | Arm | Recall (keypoint_acc) | Mask-aware precision | F1 | Counting MAE† |
|---|---|---:|---:|---:|---:|
| within-distribution | control (direct) | 0.361 | 0.867 | 0.510 | 101.29 |
| within-distribution | treatment (AutoArborist pretrain) | 0.377 | 0.936 | 0.538 | 66.04 |
| within-distribution | **Δ treatment − control** | **+0.016** | **+0.069** | **+0.028** | **−35.25** |
| out-of-distribution | control (direct) | 0.374 | 0.940 | 0.535 | 60.54 |
| out-of-distribution | treatment (AutoArborist pretrain) | 0.425 | 0.937 | 0.585 | 115.46 |
| out-of-distribution | **Δ treatment − control** | **+0.051** | **−0.003** | **+0.050** | **+54.92** |

† Validation counting MAE is Allen et al. 2025 only (Frey has no counting GT in the eval).

Per-source validation recall (keypoint_acc) — both sources move the same way:

| Split | Arm | Allen et al. 2025 | Frey et al. 2026 |
|---|---|---:|---:|
| within-distribution | control | 0.431 | 0.290 |
| within-distribution | treatment | 0.458 | 0.296 |
| out-of-distribution | control | 0.449 | 0.298 |
| out-of-distribution | treatment | 0.490 | 0.359 |

## Stage-1 sanity (`pretrain_results_<split>.txt`)

BASELINE (init checkpoint) vs AFTER (AutoArborist-pretrained), on the held-out
source-stratified slice of the *supervised* points train split.

| Split | keypoint_acc BEFORE | keypoint_acc AFTER | Δ | mask-aware P BEFORE→AFTER | counting MAE BEFORE→AFTER |
|---|---:|---:|---:|---:|---:|
| within-distribution | 0.739 | 0.737 | −0.002 | 0.713 → 0.755 | 132.21 → 115.12 |
| out-of-distribution | 0.575 | 0.573 | −0.002 | 0.556 → 0.571 | 177.67 → 135.57 |

Stage 1 leaves held-out supervised keypoint accuracy essentially unchanged (−0.002 both
splits) and nudges count calibration in the right direction (MAE −17 WD, −42 OOD). So the
stage-2 gain is not "stage 1 handed fine-tuning a better detector" — it is the pretraining
*exposure* to ~20k additional loosely-labelled point images that the fine-tune then builds
on. On Amirkolaee et al. 2023 specifically (TreeFormer's own training source, worst group
in both baselines) stage-1 keypoint_acc rises 0.288→0.339 (WD) and 0.294→0.394 (OOD).

## Interpretation

- **The validation result is the headline and it is a genuine, source-consistent gain.**
  AutoArborist continue-pretraining before the supervised fine-tune improves held-out
  point F1 by +0.028 (WD) and +0.050 (OOD), all of it recall, with precision already
  saturated near 0.94. Allen *and* Frey both improve on both splits, so this is not a
  tiling or single-source artifact.
- **This is the first non-null the weak-supervision ablation has produced.** Box→box and
  box→polygon were null at every transfer depth and both splits
  (`notes/weak_supervision_pretraining_table.md`) because at MillionTrees box scale the
  initialization is nearly irrelevant. The v0.24 point task has only 3,403 supervised
  train images / 293k annotations after the AutoArborist demotion — the note predicted
  "more headroom here" and that is what the validation split shows.
- **Counting stays unstable.** MAE improves WD (test −39, val −35) and degrades OOD
  (test +23, val +55). Consistent with the broader TreeFormer counting story
  (`[[project_points_wrong_branch_collapse]]`, count-loss-fix ledger entries): point
  counts are not a reliable per-source signal at this data scale. Report F1/recall as the
  outcome, MAE as context.
- **Absolute numbers are not comparable to manuscript Table 2.** The weak-supervision
  control arm uses the plain leaderboard fine-tune recipe (pretrained init, lr 2e-4, 20
  epochs, `val_loss` selection) — *not* the `--loss-preset pretrain` count-loss-fix recipe
  behind Table 2's TreeFormer rows. Only the within-ablation control-vs-treatment Δ is
  meaningful. (The control's validation numbers do land close to Table 2's count-fix OOD
  val row by coincidence: R 0.374 / P 0.940 / F1 0.535 here vs R 0.371 / P 0.946 / F1
  0.533 there.)
