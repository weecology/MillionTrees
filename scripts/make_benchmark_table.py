"""Generate benchmark comparison tables from training and existing-model evaluation results.

Parses the metrics out of the result .txt files written by the training and
existing_models eval scripts and rewrites the generated section of
docs/leaderboard.md: one metrics table per split, plus a run-configuration table
recording what produced each number.

Every published row is declared explicitly in ``RUNS`` below, because the
leaderboard run for a geometry is frequently NOT the one in the default output
directory — the points rows come from ``<split>_896_countfix``, the polygon rows
from ``detectron2/<split>``, and the stale default directories still hold older
experiments that would otherwise be published silently. Adding a row here is the
only way a model reaches the leaderboard, and the ``config`` fields are published
alongside the metrics so a reader can reproduce the number.
"""

import argparse
import os
import re
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Metric keys to extract (lowercase prefix matched against results_str lines).
# Box/point files emit geometry-specific headline lines (detection_accuracy,
# keypointaccuracy); every geometry also emits the generic "Average accuracy/
# recall/AP40" block, which for polygons is the mask accuracy/recall.
METRIC_PATTERNS = {
    "KeypointAccuracy": r"average (?:keypointaccuracy|keypoint_acc across source):\s*([\d.]+)",
    "CountingMAE": r"average counting_mae:\s*([\d.]+)",
    "DetectionAccuracy": r"average detection_accuracy:\s*([\d.]+)",
    "DetectionRecall": r"average detection_recall:\s*([\d.]+)",
    "MaskAwarePrecision": r"average maskaware_precision:\s*([\d.]+)",
    "MaskAccuracy": r"average accuracy:\s*([\d.]+)",
    "MaskRecall": r"average recall:\s*([\d.]+)",
    # AP at IoU 0.4, the same match threshold used by recall / mask-aware
    # precision (and therefore F1). Runs made before AP moved off IoU 0.5 only
    # report AP50 and show "-" here until they are re-evaluated.
    "AP40": r"average ap40:\s*([\d.]+)",
}

# For each task, the (recall, precision) metrics used to compute the detection F1.
# Recall is geometry-specific; precision is the mask-aware precision across all
# geometries. F1 is the harmonic mean of the two.
F1_INPUTS = {
    "TreeBoxes": ("DetectionRecall", "MaskAwarePrecision"),
    "TreePoints": ("KeypointAccuracy", "MaskAwarePrecision"),
    "TreePolygons": ("MaskRecall", "MaskAwarePrecision"),
}

# Map task names to which metrics are relevant (recall, precision, F1 first so the
# F1 sort key is easy to read against its inputs).
TASK_METRICS = {
    "TreeBoxes": ["DetectionRecall", "MaskAwarePrecision", "F1", "AP40", "CountingMAE"],
    "TreePoints": ["KeypointAccuracy", "MaskAwarePrecision", "F1", "CountingMAE"],
    "TreePolygons": ["MaskRecall", "MaskAwarePrecision", "F1", "MaskAccuracy", "AP40"],
}

# Dataset version every published row is scored against. Bump on a release, and
# re-run every row -- mixing versions across rows makes the table uncomparable.
DATA_VERSION = "v0.23"

# The published leaderboard rows.
#
#   model      display name
#   task       TreeBoxes | TreePoints | TreePolygons
#   finetuned  trained on the MillionTrees train split
#   path       results file, {split} substituted; relative to the repo root
#   splits     which split schemes this row is published for
#   config     what a reader needs to reproduce the number:
#              weights, score threshold, eval image size, environment, and the
#              training hyperparameters for fine-tuned rows
RUNS = [
    # ---- fine-tuned on MillionTrees ------------------------------------------
    {
        "model": "DeepForest (RetinaNet)",
        "task": "TreeBoxes",
        "finetuned": True,
        "path": "training/boxes/outputs/{split}/results_{split}.txt",
        "splits": ["within-distribution", "out-of-distribution"],
        "config": {
            "weights": "trained from the DeepForest release backbone",
            "score_threshold": "0.10",
            "eval_image_size": "native tile",
            "env": "shared `.venv` (`uv run`)",
            "train": ("`training/slurm/train_boxes.sbatch`: batch 32 x 2 GPUs, lr 0.01, "
                      "<=200 epochs, early-stop patience 10 (within) / 15 (OOD), "
                      "no gradient clipping, no seed set"),
        },
    },
    {
        "model": "TreeFormer (count-loss fix)",
        "task": "TreePoints",
        "finetuned": True,
        "path": "training/points/outputs/{split}_896_countfix/results_{split}.txt",
        "splits": ["within-distribution", "out-of-distribution"],
        "config": {
            "weights": "`weecology/deepforest-tree-point`, fine-tuned (job 39014685)",
            "score_threshold": "0.10 (`score_integration_radius` 2)",
            "eval_image_size": "896",
            "env": "frozen `.venv-treeformer` (DeepForest `treeformer-training` branch)",
            "train": ("`training/slurm/train_points_896_countfix.sbatch`: batch 16 x 3 GPUs, "
                      "lr 2e-4, 20 epochs, image 896, `--loss-preset pretrain` "
                      "(enforce_count False, losses count/ot/density_l1, mae_weight 0.025, "
                      "density_l1_weight 0.05), checkpoint on val_loss, early stopping off, "
                      "no seed set"),
        },
    },
    {
        "model": "Mask R-CNN (Detectron2)",
        "task": "TreePolygons",
        "finetuned": True,
        "path": "training/polygons/outputs/detectron2/{split}/results_{split}.txt",
        "splits": ["within-distribution", "out-of-distribution"],
        "config": {
            "weights": "COCO Mask R-CNN R50-FPN 3x; checkpoint `model_final.pth`",
            "score_threshold": "0.15 (`ROI_HEADS.SCORE_THRESH_TEST`)",
            "eval_image_size": "448 (`min-size-test` 800, `max-size` 1333)",
            "env": "CanopyRS uv venv (from-source Detectron2)",
            "train": ("`training/slurm/train_polygons_detectron2.sbatch`: batch 8, lr 0.01, "
                      "50 epochs, image 448, `min-size-train` 640-800, LR steps at 70%/90% "
                      "of iters, warmup <=1000 iters, seed 42"),
        },
    },
    {
        "model": "DeepForest Mask R-CNN",
        "task": "TreePolygons",
        "finetuned": True,
        "path": ("training/polygons/outputs/deepforest_annotationsafecrop/{split}/"
                 "results_{split}.txt"),
        "splits": ["within-distribution", "out-of-distribution"],
        "config": {
            "weights": "torchvision Mask R-CNN, COCO init",
            "score_threshold": "0.10",
            "eval_image_size": "448, tiled stream eval",
            "env": "shared `.venv` (`uv run`)",
            "train": ("`training/slurm/train_polygons_deepforest_annotationsafecrop.sbatch`: "
                      "batch 16, lr 0.01, <=100 epochs, image 448, "
                      "`--train-aug annotationsafecrop`, `--eval-inference tiled`, "
                      "`--data-scope subset`, seed 42"),
        },
    },
    # ---- pretrained / zero-shot ----------------------------------------------
    {
        "model": "CanopyRS DINO Swin-L",
        "task": "TreeBoxes",
        "finetuned": False,
        "path": "existing_models/canopyrs/outputs/{split}/results_boxes_{split}.txt",
        "splits": ["within-distribution", "out-of-distribution"],
        "config": {
            "weights": "CanopyRS DINO Swin-L release",
            "score_threshold": "**0.30** (per-model tuned; see `docs/canopyrs_threshold_sweep.md`)",
            "eval_image_size": "CanopyRS default tiling",
            "env": "`existing_models/canopyrs/.venv`",
            "train": "not trained on MillionTrees",
        },
    },
    {
        "model": "DeepForest (release weights)",
        "task": "TreeBoxes",
        "finetuned": False,
        "path": "existing_models/deepforest/outputs/{split}/results_boxes_{split}.txt",
        "splits": ["within-distribution", "out-of-distribution"],
        "config": {
            "weights": "DeepForest release weights",
            "score_threshold": "0.10",
            "eval_image_size": "native tile",
            "env": "deepforest venv",
            "train": "not trained on MillionTrees",
        },
    },
    {
        "model": "TreeFormer (release weights)",
        "task": "TreePoints",
        "finetuned": False,
        "path": ("existing_models/treeformer/outputs/{split}_896_isolated/"
                 "results_points_{split}.txt"),
        "splits": ["within-distribution", "out-of-distribution"],
        "config": {
            "weights": "`weecology/deepforest-tree-point`",
            "score_threshold": "0.10 (`score_integration_radius` 2)",
            "eval_image_size": "896",
            "env": "frozen `.venv-treeformer`",
            "train": "not trained on MillionTrees",
        },
    },
    {
        "model": "CanopyRS DINO + SAM3 (SelvaMask)",
        "task": "TreePolygons",
        "finetuned": False,
        "path": "existing_models/canopyrs/outputs/{split}/results_polygons_{split}.txt",
        "splits": ["within-distribution", "out-of-distribution"],
        "config": {
            "weights": "CanopyRS DINO detector + SAM3 SelvaMask segmenter",
            "score_threshold": "**0.30** (per-model tuned)",
            "eval_image_size": "CanopyRS default tiling",
            "env": "`existing_models/canopyrs/.venv`",
            "train": "not trained on MillionTrees",
        },
    },
    {
        "model": "detectree2",
        "task": "TreePolygons",
        "finetuned": False,
        "path": "existing_models/detectree2/outputs/{split}/results_polygons_{split}.txt",
        "splits": ["within-distribution", "out-of-distribution"],
        "config": {
            "weights": "`250312_flexi.pth`",
            "score_threshold": "0.10",
            "eval_image_size": "detectree2 default tiling",
            "env": "detectree2 venv",
            "train": "not trained on MillionTrees",
        },
    },
    {
        "model": "SAM3",
        "task": "TreeBoxes",
        "finetuned": False,
        "path": "existing_models/sam3/outputs/{split}/results_boxes_{split}.txt",
        "splits": ["within-distribution", "out-of-distribution"],
        "config": {
            "weights": "`facebook/sam3`",
            "score_threshold": "0.10",
            "eval_image_size": "SAM3 default",
            "env": "sam3 venv",
            "train": "not trained on MillionTrees",
        },
    },
    {
        "model": "SAM3",
        "task": "TreePoints",
        "finetuned": False,
        "path": "existing_models/sam3/outputs/{split}/results_points_{split}.txt",
        "splits": ["within-distribution", "out-of-distribution"],
        "config": {
            "weights": "`facebook/sam3`",
            "score_threshold": "0.10",
            "eval_image_size": "SAM3 default",
            "env": "sam3 venv",
            "train": "not trained on MillionTrees",
        },
    },
    {
        "model": "SAM3",
        "task": "TreePolygons",
        "finetuned": False,
        "path": "existing_models/sam3/outputs/{split}/results_polygons_{split}.txt",
        "splits": ["within-distribution", "out-of-distribution"],
        "config": {
            "weights": "`facebook/sam3`",
            "score_threshold": "0.10",
            "eval_image_size": "SAM3 default",
            "env": "sam3 venv",
            "train": "not trained on MillionTrees",
        },
    },
    # ---- cross-geometry (polygons predicted from another geometry) -----------
    {
        "model": "TreeFormer + SAM2",
        "task": "TreePolygons",
        "finetuned": False,
        "path": ("existing_models/treeformer_sam2/outputs/crossgeometry_v023/"
                 "results_polygons_crossgeometry.txt"),
        "splits": ["crossgeometry"],
        "config": {
            "weights": "TreeFormer points -> SAM2 mask prompting",
            "score_threshold": "0.10",
            "eval_image_size": "896 (points stage)",
            "env": "frozen `.venv-treeformer` + SAM2",
            "train": "not trained on MillionTrees",
        },
    },
]


def parse_metrics(text: str) -> Dict[str, float]:
    metrics = {}
    lower = text.lower()
    for name, pattern in METRIC_PATTERNS.items():
        m = re.search(pattern, lower)
        if m:
            metrics[name] = float(m.group(1))
    return metrics


def add_f1(task: str, metrics: Dict[str, float]) -> None:
    """Add the detection F1 (harmonic mean of recall and mask-aware precision)
    to ``metrics`` in place, when both inputs are present and positive."""
    recall_key, precision_key = F1_INPUTS.get(task, (None, None))
    recall = metrics.get(recall_key)
    precision = metrics.get(precision_key)
    if recall is not None and precision is not None and (recall + precision) > 0:
        metrics["F1"] = 2 * recall * precision / (recall + precision)


def collect_entries(split: str) -> List[Dict]:
    """Read every registered run that publishes a row for ``split``.

    A registered run whose results file is missing is reported to stdout rather
    than skipped silently -- a missing file means the leaderboard is
    incomplete, not that the model does not compete.
    """
    entries = []
    for run in RUNS:
        if split not in run["splits"]:
            continue
        path = os.path.join(ROOT, run["path"].format(split=split))
        if not os.path.isfile(path):
            print(f"  MISSING {run['model']} ({run['task']}): {run['path'].format(split=split)}")
            continue
        with open(path) as f:
            metrics = parse_metrics(f.read())
        if not metrics:
            print(f"  NO METRICS PARSED {run['model']} ({run['task']}): {path}")
            continue
        add_f1(run["task"], metrics)
        entries.append({
            "model": run["model"],
            "task": run["task"],
            "finetuned": run["finetuned"],
            "config": run["config"],
            "split": split,
            "metrics": metrics,
            "source": os.path.relpath(path, ROOT),
        })
    return entries


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def build_table(entries: List[Dict], split: str) -> str:
    if not entries:
        return f"*No results found for {split} split.*\n"

    lines = []
    for task in sorted({e["task"] for e in entries}):
        task_entries = [e for e in entries if e["task"] == task]

        metric_names = TASK_METRICS.get(task, list(METRIC_PATTERNS.keys()))
        # Only show metrics that appear in at least one entry
        present_metrics = [m for m in metric_names
                           if any(m in e["metrics"] for e in task_entries)]
        if not present_metrics:
            present_metrics = sorted({k for e in task_entries for k in e["metrics"]})

        header = "| Model | Fine-tuned | " + " | ".join(present_metrics) + " |\n"
        sep = "|---" * (len(present_metrics) + 2) + "|\n"
        rows = []

        # Sort by F1 descending (entries without an F1 fall to the bottom),
        # breaking ties by model name for stable output.
        def sort_key(x):
            f1 = x["metrics"].get("F1")
            return (f1 is None, -(f1 or 0.0), x["model"])

        for e in sorted(task_entries, key=sort_key):
            vals = " | ".join(format_metric(e["metrics"].get(m)) for m in present_metrics)
            ft = "✓" if e["finetuned"] else "✗"
            rows.append(f"| {e['model']} | {ft} | {vals} |")

        lines.append(f"### {task}\n\n{header}{sep}" + "\n".join(rows) + "\n")

    return "\n".join(lines)


def build_config_table(all_entries: List[Dict]) -> str:
    """One row per published model/geometry: the configuration behind its score.

    Deduplicated across splits -- a model's config is identical on both splits
    except where the run registry says otherwise.
    """
    seen = set()
    rows = []
    for e in sorted(all_entries, key=lambda x: (x["task"], not x["finetuned"], x["model"])):
        key = (e["model"], e["task"])
        if key in seen:
            continue
        seen.add(key)
        c = e["config"]
        ft = "✓" if e["finetuned"] else "✗"
        rows.append(
            f"| {e['model']} | {e['task']} | {ft} | {c['weights']} | {c['score_threshold']} | "
            f"{c['eval_image_size']} | {c['env']} | {c['train']} |"
        )

    header = ("| Model | Geometry | Fine-tuned | Weights / checkpoint | Score threshold | "
              "Eval image size | Environment | Training configuration |\n")
    sep = "|---" * 8 + "|\n"
    return header + sep + "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison tables.")
    parser.add_argument("--splits", nargs="+",
                        default=["within-distribution", "out-of-distribution", "crossgeometry"],
                        help="Split schemes to include")
    parser.add_argument("--output", type=str,
                        default=os.path.join(ROOT, "docs", "leaderboard.md"),
                        help="Output markdown file to update")
    args = parser.parse_args()

    all_entries_by_split: Dict[str, List[Dict]] = {}
    for split in args.splits:
        print(f"Split '{split}':")
        entries = collect_entries(split)
        all_entries_by_split[split] = entries
        print(f"  found {len(entries)} of "
              f"{sum(split in r['splits'] for r in RUNS)} registered result file(s)")

    section = "## Benchmark Results\n\n"
    section += (f"Fine-tuned models (trained on the MillionTrees train split) vs. pretrained "
                f"models evaluated zero-shot, all on MillionTrees **{DATA_VERSION}**. "
                f"All AP is **AP40** (IoU 0.4), the same match threshold behind recall and "
                f"mask-aware precision; F1 is their harmonic mean. Rows are generated from "
                f"the result files by `scripts/make_benchmark_table.py` -- see the run "
                f"configuration table below for what produced each number.\n\n")

    for split in args.splits:
        entries = all_entries_by_split[split]
        section += f"### Split: {split}\n\n"
        section += build_table(entries, split) + "\n"

    section += "## Run configuration\n\n"
    section += ("Everything needed to reproduce a row beyond the dataset version. Score "
                "thresholds are the standardized 0.10 unless marked otherwise.\n\n")
    section += build_config_table(
        [e for entries in all_entries_by_split.values() for e in entries])

    marker = "## Benchmark Results"
    if os.path.isfile(args.output):
        with open(args.output, encoding="utf-8") as f:
            md = f.read()
        start = md.find(marker)
        if start != -1:
            md = md[:start] + section
        else:
            md = md.rstrip() + "\n\n" + section
    else:
        md = section

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote benchmark tables to {args.output}")


if __name__ == "__main__":
    main()
