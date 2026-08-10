"""Plot the CanopyRS score-threshold sweep as a precision-recall curve.

Reads the per-split threshold_sweep.csv files produced by
existing_models/slurm/sweep_canopyrs_pr_curve.sbatch (which drives
existing_models/canopyrs/eval_{boxes,polygons}.py --sweep on the manuscript
splits) and renders one PR curve per (task, split), plus a companion
recall/precision/F1-vs-threshold panel that motivates the recommended
operating point. Also writes the best-F1 threshold per (task, split) as a
markdown table fragment for docs/canopyrs_threshold_sweep.md.
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO = "/blue/ewhite/b.weinstein/src/MillionTrees"
SWEEP_GLOB = os.path.join(REPO, "existing_models/canopyrs/outputs/sweep/*/threshold_sweep.csv")
OUT_PNG = os.path.join(REPO, "docs/public/canopyrs_threshold_sweep_pr_curve.png")
OUT_TABLE_MD = os.path.join(REPO, "docs/canopyrs_threshold_sweep_table.md")

# geometry -> color (categorical, fixed order), split -> linestyle (secondary encoding)
TASK_COLOR = {
    "TreeBoxes": "#2a78d6",     # blue
    "TreePolygons": "#e34948",  # red
}
SPLIT_STYLE = {
    "within-distribution": {"linestyle": "-", "marker": "o"},
    "out-of-distribution": {"linestyle": "--", "marker": "s"},
}
LABEL_THRESHOLDS = {0.1, 0.3, 0.5, 0.7}
TASK_LABEL = {"TreeBoxes": "Boxes", "TreePolygons": "Polygons"}
SPLIT_LABEL = {"within-distribution": "Within-distribution", "out-of-distribution": "Out-of-distribution"}


def load_sweep():
    csvs = sorted(glob.glob(SWEEP_GLOB))
    if not csvs:
        raise SystemExit(f"No threshold_sweep.csv found under {SWEEP_GLOB}")
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df = df[df["model"].str.startswith("CanopyRS")]
    df = df.sort_values(["task", "split", "threshold"])
    return df


def plot_pr_curve(df, ax):
    ax.set_facecolor("#fcfcfb")
    for (task, split), g in df.groupby(["task", "split"]):
        g = g.sort_values("threshold")
        style = SPLIT_STYLE[split]
        ax.plot(g["recall"], g["precision"], color=TASK_COLOR[task],
                linestyle=style["linestyle"], marker=style["marker"],
                markersize=6, linewidth=2,
                label=f"{TASK_LABEL[task]} / {SPLIT_LABEL[split]}")
        for _, row in g.iterrows():
            if row["threshold"] in LABEL_THRESHOLDS:
                ax.annotate(f"{row['threshold']:.2g}", (row["recall"], row["precision"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7.5,
                            color="#52514e")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("CanopyRS precision-recall curve across score_threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.5, color="#e5e4df", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")


def plot_f1_vs_threshold(df, ax):
    ax.set_facecolor("#fcfcfb")
    for (task, split), g in df.groupby(["task", "split"]):
        g = g.sort_values("threshold")
        style = SPLIT_STYLE[split]
        ax.plot(g["threshold"], g["f1"], color=TASK_COLOR[task],
                linestyle=style["linestyle"], marker=style["marker"],
                markersize=6, linewidth=2,
                label=f"{TASK_LABEL[task]} / {SPLIT_LABEL[split]}")
        best = g.loc[g["f1"].idxmax()]
        ax.scatter([best["threshold"]], [best["f1"]], s=60, facecolors="none",
                   edgecolors=TASK_COLOR[task], linewidths=1.5, zorder=5)
    ax.set_xlabel("score_threshold")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs. score_threshold (circle = best F1)")
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.5, color="#e5e4df", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")


def write_best_table(df):
    valid = df[df["f1"] > 0]
    best = valid.loc[valid.groupby(["task", "split"])["f1"].idxmax()]
    best = best.sort_values(["task", "split"])
    lines = ["| Task | Split | Best threshold | Recall | Precision | F1 |",
             "|---|---|---:|---:|---:|---:|"]
    for _, r in best.iterrows():
        lines.append(f"| {TASK_LABEL[r.task]} | {SPLIT_LABEL[r.split]} | {r.threshold:.2f} | "
                     f"{r.recall:.3f} | {r.precision:.3f} | {r.f1:.3f} |")
    with open(OUT_TABLE_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT_TABLE_MD}")


def main():
    df = load_sweep()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
    plot_pr_curve(df, axes[0])
    plot_f1_vs_threshold(df, axes[1])
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, facecolor="#fcfcfb")
    print(f"Wrote {OUT_PNG}")
    write_best_table(df)


if __name__ == "__main__":
    main()
