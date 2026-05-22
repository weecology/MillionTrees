# Weak-supervision pretraining experiment

Compare polygon Mask R-CNN performance under:

- `coco` — full model pretrained on MS COCO (torchvision `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1`; backbone + FPN + heads)
- `box_pretrained` — backbone weights transferred from TreeBoxes DeepForest pretraining; heads randomly initialized

## Scripts

| Script | Purpose |
| --- | --- |
| `training/boxes/pretrain_backbone_for_polygons.py` | Train/finetune on TreeBoxes; export `box_backbone_<split>.pt` + JSON metadata |
| `training/boxes/audit_weak_supervision_volume.py` | CSV-level counts of weak-supervision rows |
| `scripts/make_weak_supervision_report.py` | Figure + markdown table from `results_*.txt` files |

## Polygon training flags (`training/polygons/train_polygons.py`)

- `--init-mode {coco,box_pretrained}`
- `--box-backbone-checkpoint PATH` (required for `box_pretrained`)
- `--include-unsupervised`, `--data-scope {subset,full}`, `--seed`

Training writes `results_<split>.txt` and `results_<split>.json` with `run_metadata`.

## Reproduce

Audit annotation volume (subset vs full weak supervision):

```bash
uv run python training/boxes/audit_weak_supervision_volume.py \
  --root-dir data \
  --split-scheme random \
  --data-scope subset \
  --output-dir training/weak_supervision/outputs/audit

uv run python training/boxes/audit_weak_supervision_volume.py \
  --root-dir data \
  --split-scheme random \
  --include-unsupervised \
  --data-scope full \
  --output-dir training/weak_supervision/outputs/audit
```

Submit SLURM jobs (backbone pretrain + both polygon tracks):

```bash
bash training/weak_supervision/submit_weak_supervision.sh
```

Generate figure and table (once all jobs complete):

```bash
uv run python scripts/make_weak_supervision_report.py \
  --results-dir training/weak_supervision/outputs \
  --figure-out docs/public/weak_supervision_pretraining_gain.png \
  --table-out docs/weak_supervision_pretraining_table.md
```

## Notes

- Full NEON-scale counts require local `TreeBoxes_v0.12` with `--include-unsupervised` and matching CSVs.
