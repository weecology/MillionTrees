"""Tabulate AP50 vs AP40 against F1 for the pretrained (existing) models.

Motivation: the leaderboard reports AP at IoU 0.5, while recall and mask-aware
precision -- and therefore F1 -- match predictions to ground truth at IoU 0.4.
One proposed explanation for the large F1-vs-AP50 gap is simply that different
IoU thresholds decide what counts as a match. The datasets now compute both
``AP50`` and ``AP40`` on the same predictions, so this script reads the eval
result files and prints recall / precision / F1 / AP50 / AP40 side by side.

Usage:
    python scripts/make_ap_iou_table.py \
        --outputs-dirname outputs_ap_iou \
        --output notes/ap50_vs_ap40_existing_models.md
"""

import argparse
import os
from typing import Dict, List, Optional

from make_benchmark_table import (F1_INPUTS, ROOT, find_existing_model_results)

# Recall column is geometry-specific; precision is mask-aware everywhere.
RECALL_LABEL = {
    "TreeBoxes": "DetectionRecall",
    "TreePoints": "KeypointAccuracy",
    "TreePolygons": "MaskRecall",
}


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def _delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "-"
    return f"{a - b:+.3f}"


def build_rows(splits: List[str], outputs_dirname: str) -> List[Dict]:
    rows = []
    for split in splits:
        for entry in find_existing_model_results(split, outputs_dirname):
            m = entry["metrics"]
            if "AP40" not in m and "AP50" not in m:
                # Points, or a result file written before AP40 existed.
                continue
            recall_key = F1_INPUTS[entry["task"]][0]
            rows.append({
                "task": entry["task"],
                "split": split,
                "model": entry["model"],
                "recall": m.get(recall_key),
                "precision": m.get("MaskAwarePrecision"),
                "f1": m.get("F1"),
                "ap50": m.get("AP50"),
                "ap40": m.get("AP40"),
                "source": entry["source"],
            })
    return rows


def build_markdown(rows: List[Dict], splits: List[str]) -> str:
    out = ["# AP50 vs AP40 (pretrained models, test split)", ""]
    out.append(
        "AP50 matches predictions to ground truth at IoU 0.5; AP40 uses IoU 0.4, "
        "the same threshold the recall / mask-aware precision behind F1 use. Both "
        "are computed on identical predictions in a single eval pass, at each "
        "model's leaderboard score threshold. `AP40-AP50` isolates how much of the "
        "F1-vs-AP gap is the IoU criterion alone.")
    out.append("")

    if not rows:
        out.append("*No result files found.*")
        return "\n".join(out) + "\n"

    for split in splits:
        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            continue
        out.append(f"## Split: {split}")
        out.append("")
        out.append("| Task | Model | Recall | Precision | F1 | AP50 | AP40 | "
                   "AP40-AP50 | F1-AP40 |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(split_rows, key=lambda x: (x["task"], x["model"])):
            out.append("| {task} | {model} | {rec} | {prec} | {f1} | {ap50} | "
                       "{ap40} | {d_ap} | {d_f1} |".format(
                           task=r["task"],
                           model=r["model"],
                           rec=_fmt(r["recall"]),
                           prec=_fmt(r["precision"]),
                           f1=_fmt(r["f1"]),
                           ap50=_fmt(r["ap50"]),
                           ap40=_fmt(r["ap40"]),
                           d_ap=_delta(r["ap40"], r["ap50"]),
                           d_f1=_delta(r["f1"], r["ap40"])))
        out.append("")

    out.append("Result files:")
    out.append("")
    for r in sorted(rows, key=lambda x: x["source"]):
        out.append(f"- `{r['source']}`")
    out.append("")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="AP50 vs AP40 comparison table for pretrained models.")
    parser.add_argument("--splits",
                        nargs="+",
                        default=["within-distribution", "out-of-distribution"])
    parser.add_argument("--outputs-dirname",
                        default="outputs_ap_iou",
                        help="Results tree under each existing_models/<model>/ "
                        "to read (default: the AP-IoU comparison runs).")
    parser.add_argument("--output",
                        default=os.path.join(ROOT, "notes",
                                             "ap50_vs_ap40_existing_models.md"))
    args = parser.parse_args()

    rows = build_rows(args.splits, args.outputs_dirname)
    md = build_markdown(rows, args.splits)
    print(md)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {args.output} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
