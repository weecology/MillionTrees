"""Generate weak-supervision comparison figure and table from results_*.txt files."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Metrics to extract from the txt summary block
METRIC_PATTERNS = {
    "accuracy": re.compile(r"^Average accuracy:\s+([\d.]+)", re.MULTILINE),
    "recall": re.compile(r"^Average recall:\s+([\d.]+)", re.MULTILINE),
    "AP50": re.compile(r"^Average AP50:\s+([\d.]+)", re.MULTILINE),
    "maskaware_precision": re.compile(r"^Average maskaware_precision:\s+([\d.]+)", re.MULTILINE),
}

PRIMARY_METRIC = "AP50"


def parse_txt(path):
    text = path.read_text(encoding="utf-8")
    metrics = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            metrics[name] = float(match.group(1))
    return metrics


def load_rows(results_dir):
    rows = []
    root = Path(results_dir)
    for txt_path in sorted(root.glob("**/results_*.txt")):
        parts = txt_path.parts
        # Infer init_mode from parent directory name
        if "polygon_coco" in parts:
            init_mode = "coco"
        elif "polygon_box_pretrained" in parts:
            init_mode = "box_pretrained"
        else:
            continue
        split = re.search(r"results_(.+)\.txt", txt_path.name)
        if not split:
            continue
        split = split.group(1)
        metrics = parse_txt(txt_path)
        if not metrics:
            continue
        rows.append({"init_mode": init_mode, "split": split, "path": str(txt_path), **metrics})
    return pd.DataFrame(rows)


def build_comparison_table(df):
    scratch = df[df["init_mode"] == "coco"].set_index("split")
    box = df[df["init_mode"] == "box_pretrained"].set_index("split")
    splits = sorted(set(scratch.index) & set(box.index))
    records = []
    for split in splits:
        row = {"split": split}
        for metric in METRIC_PATTERNS:
            s = scratch.loc[split, metric] if metric in scratch.columns else float("nan")
            b = box.loc[split, metric] if metric in box.columns else float("nan")
            row[f"coco_{metric}"] = s
            row[f"box_{metric}"] = b
            row[f"delta_{metric}"] = b - s
        records.append(row)
    return pd.DataFrame(records)


def save_plot(table_df, figure_out):
    metrics = list(METRIC_PATTERNS.keys())
    splits = table_df["split"].tolist()
    n_metrics = len(metrics)
    n_splits = len(splits)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        x = range(n_splits)
        width = 0.38
        coco_vals = [table_df.loc[i, f"coco_{metric}"] for i in range(n_splits)]
        box_vals = [table_df.loc[i, f"box_{metric}"] for i in range(n_splits)]
        ax.bar([i - width / 2 for i in x], coco_vals, width=width, label="coco")
        ax.bar([i + width / 2 for i in x], box_vals, width=width, label="box_pretrained")
        for i in range(n_splits):
            delta = table_df.loc[i, f"delta_{metric}"]
            ymax = max(coco_vals[i], box_vals[i])
            ax.text(i, ymax + 0.005, f"{delta:+.3f}", ha="center", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(splits, rotation=20, ha="right")
        ax.set_title(metric)
        ax.legend(fontsize=7)

    fig.suptitle("Polygon performance: COCO vs box-pretrained backbone", y=1.02)
    fig.tight_layout()
    Path(figure_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_markdown_table(table_df, table_out):
    metrics = list(METRIC_PATTERNS.keys())
    metric_headers = " | ".join(
        f"coco {m} | box_pretrained {m} | delta {m}" for m in metrics
    )
    sep = " | ".join(["---:"] * (3 * len(metrics)))
    lines = [
        f"| split | {metric_headers} |",
        f"|---|{sep}|",
    ]
    for row in table_df.itertuples():
        cells = [row.split]
        for m in metrics:
            s = getattr(row, f"coco_{m}", float("nan"))
            b = getattr(row, f"box_{m}", float("nan"))
            d = getattr(row, f"delta_{m}", float("nan"))
            cells += [f"{s:.4f}", f"{b:.4f}", f"{d:+.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    Path(table_out).parent.mkdir(parents=True, exist_ok=True)
    Path(table_out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Make weak-supervision figure + table.")
    parser.add_argument("--results-dir", type=str, default="training/weak_supervision/outputs")
    parser.add_argument("--figure-out", type=str,
                        default="docs/public/weak_supervision_pretraining_gain.png")
    parser.add_argument("--table-out", type=str,
                        default="docs/weak_supervision_pretraining_table.md")
    parser.add_argument("--csv-out", type=str,
                        default="training/weak_supervision/outputs/weak_supervision_table.csv")
    args = parser.parse_args()

    df = load_rows(args.results_dir)
    if df.empty:
        print("No results_*.txt files found — run polygon training jobs first.")
        return

    print(f"Loaded {len(df)} result files:\n{df[['init_mode','split'] + list(METRIC_PATTERNS)].to_string(index=False)}\n")

    table_df = build_comparison_table(df)
    if table_df.empty:
        print("No paired coco + box_pretrained results for the same split found.")
        return

    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(args.csv_out, index=False)
    save_plot(table_df, args.figure_out)
    save_markdown_table(table_df, args.table_out)

    print(f"Wrote figure  -> {args.figure_out}")
    print(f"Wrote table   -> {args.table_out}")
    print(f"Wrote CSV     -> {args.csv_out}")


if __name__ == "__main__":
    main()
