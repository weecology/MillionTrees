## Torchvision Faster R-CNN on TreeBoxes

This example uses an off‑the‑shelf COCO‑pretrained Faster R‑CNN from torchvision to perform zero‑shot inference on the TreeBoxes dataset and evaluate with MillionTrees.

### Prerequisites

- Install the package and dependencies:

```bash
pip install milliontrees torch torchvision
```

- Download the TreeBoxes data if needed (or set `download=True` in code):

```python
from milliontrees import get_dataset
ds = get_dataset("TreeBoxes", version="0.2", root_dir="data", download=True)
```

### Script

Run the example script which performs inference and evaluation end‑to‑end:

```bash
python docs/examples/torchvision_fasterrcnn_treeboxes.py \
  --root_dir data \
  --version 0.2 \
  --batch_size 8 \
  --device auto
```

It prints a metrics dictionary and a formatted per‑source breakdown, including `detection_acc_avg_dom`.

### Notes

- No training is performed; this is a pure zero‑shot baseline.
- The MillionTrees evaluation expects per‑image dict predictions with keys `y`, `labels`, and `scores`. The example script handles this conversion for torchvision outputs.

