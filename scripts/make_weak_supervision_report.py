"""Generate weak-supervision comparison figure and table from experiment JSON files."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PRIMARY_METRICS = [
    "detection_acc_avg_dom",
    "average_detection_accuracy",
    "average_detection_recall",
    "average_maskaware_precision",
    "average_mAP",
]


def load_rows(results_dir):
    rows = []
    for path in sorted(Path(results_dir).glob("**/results_*.json")):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        meta = payload.get("run_metadata", {})
        row = {
            "path": str(path),
            "model": payload.get("model"),
            "task": payload.get("task"),
            "split": payload.get("split"),
            "init_mode": meta.get("init_mode", "unknown"),
            "data_scope": meta.get("data_scope", "unknown"),
            "seed": meta.get("seed"),
        }
        row.update(payload.get("metrics", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def choose_metric(df):
    for metric in PRIMARY_METRICS:
        if metric in df.columns and df[metric].notna().any():
            return metric
    raise ValueError("No primary metric found in results JSON files.")


def build_comparison_table(df, metric):
    filtered = df[df["task"] == "TreePolygons"].copy()
    base = filtered[filtered["init_mode"] == "scratch"][
        ["split", "data_scope", metric]
    ].rename(columns={metric: "scratch"})
    compare = filtered[filtered["init_mode"] == "box_pretrained"][
        ["split", "data_scope", metric]
    ].rename(columns={metric: "box_pretrained"})
    merged = base.merge(compare, on=["split", "data_scope"], how="inner")
    merged["delta"] = merged["box_pretrained"] - merged["scratch"]
    merged["delta_pct"] = (merged["delta"] / merged["scratch"]) * 100.0
    merged = merged.sort_values(["data_scope", "split"]).reset_index(drop=True)
    return merged


def save_plot(df, metric, output_path):
    labels = [f"{r.data_scope}-{r.split}" for r in df.itertuples()]
    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], df["scratch"], width=width, label="scratch")
    ax.bar([i + width / 2 for i in x], df["box_pretrained"], width=width,
           label="box_pretrained")
    for i, delta in enumerate(df["delta"]):
        ymax = max(df.loc[i, "scratch"], df.loc[i, "box_pretrained"])
        ax.text(i, ymax + 0.005, f"{delta:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title("Polygon performance: scratch vs box-pretrained backbone")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_markdown_table(df, metric, output_path):
    headers = ["Data Scope", "Split", "Scratch", "Box Pretrained", "Delta", "Delta %"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in df.itertuples():
        lines.append(
            f"| {row.data_scope} | {row.split} | {row.scratch:.4f} | "
            f"{row.box_pretrained:.4f} | {row.delta:+.4f} | {row.delta_pct:+.2f}% |"
        )
    lines.append("")
    lines.append(f"Metric used: `{metric}`")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Make weak-supervision figure + table.")
    parser.add_argument("--results-dir", type=str, default="training/weak_supervision/outputs")
    parser.add_argument("--figure-out", type=str,
                        default="docs/public/weak_supervision_pretraining_gain.png")
    parser.add_argument("--table-out", type=str,
                        default="docs/weak_supervision_pretraining_table.md")
    parser.add_argument("--csv-out", type=str,
                        default="training/weak_supervision/outputs/weak_supervision_table.csv")
    parser.add_argument(
        "--matrix-report",
        type=str,
        default="training/weak_supervision/outputs/experiment_matrix_report.json",
    )
    args = parser.parse_args()

    df = load_rows(args.results_dir)
    Path(args.figure_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.table_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)

    if not df.empty:
        metric = choose_metric(df)
        table_df = build_comparison_table(df, metric)
        if table_df.empty:
            raise ValueError("No paired scratch vs box_pretrained rows found.")
        save_plot(table_df, metric, args.figure_out)
        table_df.to_csv(args.csv_out, index=False)
        save_markdown_table(table_df, metric, args.table_out)
    else:
        with open(args.matrix_report, encoding="utf-8") as f:
            matrix = json.load(f)
        rows = []
        for track in matrix.get("tracks", []):
            rows.append({
                "data_scope": track["name"],
                "split": matrix.get("split_scheme", "unknown"),
                "scratch": None,
                "box_pretrained": None,
                "delta": None,
                "delta_pct": None,
                "scratch_returncode": track["polygon_scratch"]["returncode"],
                "box_pretrained_returncode": track["polygon_box_pretrained"]["returncode"],
            })
        table_df = pd.DataFrame(rows)
        table_df.to_csv(args.csv_out, index=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        labels = table_df["data_scope"].tolist()
        scratch_rc = table_df["scratch_returncode"].tolist()
        pre_rc = table_df["box_pretrained_returncode"].tolist()
        x_pos = range(len(labels))
        ax.bar([i - 0.18 for i in x_pos], scratch_rc, width=0.35,
               label="scratch returncode")
        ax.bar([i + 0.18 for i in x_pos], pre_rc, width=0.35,
               label="box_pretrained returncode")
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels)
        ax.set_ylabel("returncode (0 = success)")
        ax.set_title("Experiment execution status (metrics unavailable)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.figure_out, dpi=200)
        plt.close(fig)

        lines = [
            "| Data Scope | Split | Scratch | Box Pretrained | Delta | Delta % | "
            "Scratch RC | Box Pretrained RC |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in table_df.itertuples():
            lines.append(
                f"| {row.data_scope} | {row.split} | n/a | n/a | n/a | n/a | "
                f"{row.scratch_returncode} | {row.box_pretrained_returncode} |"
            )
        lines.append("")
        lines.append("No paired polygon metric JSON files were produced in this environment.")
        Path(args.table_out).write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote figure to {args.figure_out}")
    print(f"Wrote table markdown to {args.table_out}")
    print(f"Wrote table csv to {args.csv_out}")


if __name__ == "__main__":
    main()
