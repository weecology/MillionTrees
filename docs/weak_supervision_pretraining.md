# Weak-supervision pretraining experiment

Compare polygon Mask R-CNN performance under:

- `scratch` — ImageNet-free initialization (`weights=None`, `weights_backbone=None`)
- `box_pretrained` — backbone weights transferred from TreeBoxes DeepForest pretraining

Default polygon initialization remains `imagenet` (torchvision pretrained weights) when omitted.

## Scripts

| Script | Purpose |
| --- | --- |
| `training/boxes/pretrain_backbone_for_polygons.py` | Train/finetune on TreeBoxes; export `box_backbone_<split>.pt` + JSON metadata |
| `training/boxes/audit_weak_supervision_volume.py` | CSV-level counts of weak-supervision rows |
| `training/weak_supervision/run_experiment_matrix.py` | Subset + optional full track (pretrain → polygon scratch vs box_pretrained) |
| `scripts/make_weak_supervision_report.py` | Figure + markdown table from `results_*.json` (or status fallback from matrix report) |

## Polygon training flags (`training/polygons/train_polygons.py`)

- `--init-mode {imagenet,scratch,box_pretrained}`
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

Run matrix (enable downloads if data not local):

```bash
uv run python training/weak_supervision/run_experiment_matrix.py \
  --python "uv run python" \
  --root-dir data-mini \
  --subset-mini \
  --download-subset \
  --run-full \
  --full-root-dir data \
  --split-scheme random \
  --output-dir training/weak_supervision/outputs
```

Generate figure and table:

```bash
uv run python scripts/make_weak_supervision_report.py \
  --results-dir training/weak_supervision/outputs \
  --figure-out docs/public/weak_supervision_pretraining_gain.png \
  --table-out docs/weak_supervision_pretraining_table.md \
  --csv-out training/weak_supervision/outputs/weak_supervision_table.csv \
  --matrix-report training/weak_supervision/outputs/experiment_matrix_report.json
```

## Notes

- Full NEON-scale counts require local `TreeBoxes_v0.12` with `--include-unsupervised` and matching CSVs.
- If polygon result JSONs are missing (download/path errors), the report script falls back to plotting subprocess return codes from `experiment_matrix_report.json`.
