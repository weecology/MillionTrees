"""Generate benchmark comparison tables from training and existing-model evaluation results.

Scans result .txt files written by training and existing_models eval scripts, parses
metrics, and writes two markdown tables (random split, zeroshot split) to docs/leaderboard.md.

Result file conventions:
  training/{boxes,points,polygons}/outputs/{split}/results_{split}.txt
  existing_models/{model}/outputs/{split}/results_{boxes,points,polygons}_{split}.txt
"""

import argparse
import os
import re
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Metric keys to extract (lowercase prefix matched against results_str lines).
# Box/point files emit geometry-specific headline lines (detection_accuracy,
# keypointaccuracy); every geometry also emits the generic "Average accuracy/
# recall/AP50" block, which for polygons is the mask accuracy/recall.
METRIC_PATTERNS = {
    "KeypointAccuracy": r"average (?:keypointaccuracy|keypoint_acc across source):\s*([\d.]+)",
    "CountingMAE": r"average counting_mae:\s*([\d.]+)",
    "DetectionAccuracy": r"average detection_accuracy:\s*([\d.]+)",
    "DetectionRecall": r"average detection_recall:\s*([\d.]+)",
    "MaskAccuracy": r"average accuracy:\s*([\d.]+)",
    "MaskRecall": r"average recall:\s*([\d.]+)",
    "AP50": r"average ap50:\s*([\d.]+)",
}

# Map task names to which metrics are relevant
TASK_METRICS = {
    "TreeBoxes": ["DetectionAccuracy", "DetectionRecall", "CountingMAE"],
    "TreePoints": ["KeypointAccuracy", "CountingMAE"],
    "TreePolygons": ["MaskAccuracy", "MaskRecall", "AP50"],
}


def parse_metrics(text: str) -> Dict[str, float]:
    metrics = {}
    lower = text.lower()
    for name, pattern in METRIC_PATTERNS.items():
        m = re.search(pattern, lower)
        if m:
            metrics[name] = float(m.group(1))
    return metrics


def find_training_results(split: str) -> List[Dict]:
    """Discover fine-tuned training result files for a given split."""
    entries = []
    # (output subdir, task name, fine-tuned model name)
    training_models = [
        ("boxes", "TreeBoxes", "DeepForest-finetuned"),
        ("points", "TreePoints", "TreeFormer-finetuned"),
        ("polygons", "TreePolygons", "MaskRCNN-finetuned"),
    ]
    for task_dir, task_name, model_name in training_models:
        path = os.path.join(ROOT, "training", task_dir, "outputs", split,
                            f"results_{split}.txt")
        if os.path.isfile(path):
            with open(path) as f:
                text = f.read()
            metrics = parse_metrics(text)
            if metrics:
                entries.append({
                    "model": model_name,
                    "task": task_name,
                    "split": split,
                    "metrics": metrics,
                    "source": os.path.relpath(path, ROOT),
                })
    return entries


def find_existing_model_results(split: str) -> List[Dict]:
    """Discover existing-model result files for a given split."""
    entries = []
    task_map = {
        "boxes": "TreeBoxes",
        "points": "TreePoints",
        "polygons": "TreePolygons",
    }
    # Each pretrained model only competes on the geometries it natively predicts.
    model_tasks = {
        ("deepforest", "DeepForest-pretrained"): ["boxes"],
        ("treeformer", "TreeFormer-pretrained"): ["points"],
        ("sam3", "SAM3"): ["boxes", "points", "polygons"],
        ("canopyrs", "CanopyRS-DINO-SwinL"): ["boxes"],
        ("canopyrs", "CanopyRS-DINO-SAM3-SelvaMask"): ["polygons"],
        ("detectree2", "Detectree2"): ["polygons"],
    }
    for (model_dir, model_name), task_keys in model_tasks.items():
        for task_key in task_keys:
            task_name = task_map[task_key]
            path = os.path.join(ROOT, "existing_models", model_dir, "outputs", split,
                                f"results_{task_key}_{split}.txt")
            if os.path.isfile(path):
                with open(path) as f:
                    text = f.read()
                metrics = parse_metrics(text)
                if metrics:
                    entries.append({
                        "model": model_name,
                        "task": task_name,
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

    # Collect all tasks present
    tasks = sorted({e["task"] for e in entries})

    # Header columns: Model | Task | metric1 | metric2 ...
    # Use a combined header that shows metric names per task block
    lines = []

    for task in tasks:
        task_entries = [e for e in entries if e["task"] == task]
        if not task_entries:
            continue

        metric_names = TASK_METRICS.get(task, list(METRIC_PATTERNS.keys()))
        # Only show metrics that appear in at least one entry
        present_metrics = [m for m in metric_names
                           if any(m in e["metrics"] for e in task_entries)]
        if not present_metrics:
            # Fall back to whatever we found
            present_metrics = sorted({k for e in task_entries for k in e["metrics"]})

        header = "| Model | " + " | ".join(present_metrics) + " |\n"
        sep = "|---|" + "|---|" * len(present_metrics) + "\n"
        rows = []
        for e in sorted(task_entries, key=lambda x: x["model"]):
            vals = " | ".join(format_metric(e["metrics"].get(m)) for m in present_metrics)
            rows.append(f"| {e['model']} | {vals} |")

        lines.append(f"### {task}\n\n{header}{sep}" + "\n".join(rows) + "\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison tables.")
    parser.add_argument("--splits", nargs="+", default=["random", "zeroshot"],
                        help="Split schemes to include")
    parser.add_argument("--output", type=str,
                        default=os.path.join(ROOT, "docs", "leaderboard.md"),
                        help="Output markdown file to update")
    args = parser.parse_args()

    all_entries_by_split: Dict[str, List[Dict]] = {}
    for split in args.splits:
        entries = find_training_results(split) + find_existing_model_results(split)
        all_entries_by_split[split] = entries
        found = len(entries)
        print(f"Split '{split}': found {found} result file(s)")

    # Build the generated section
    section = "## Benchmark Results\n\n"
    section += ("Comparison of fine-tuned models (trained on MillionTrees) vs. "
                "pretrained models evaluated zero-shot.\n\n")

    for split in args.splits:
        entries = all_entries_by_split[split]
        section += f"### Split: {split}\n\n"
        section += build_table(entries, split) + "\n"

    # Update leaderboard.md
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
