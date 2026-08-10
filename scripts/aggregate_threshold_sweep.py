"""Aggregate per-model threshold_sweep.csv files into one comparison table.

Reads existing_models/*/outputs/sweep/threshold_sweep.csv, concatenates them, and
prints (a) the full recall/precision/AP/F1 grid per model and (b) the best-F1
threshold per (model, task, split). Writes a combined CSV + markdown summary to docs/.

The AP column is ``ap40`` (the current metric); sweeps written before AP moved to
IoU 0.4 carry ``ap50`` instead and are labelled as such.
"""

import glob
import os

import pandas as pd

REPO = "/blue/ewhite/b.weinstein/src/MillionTrees"
CSVS = glob.glob(os.path.join(REPO, "existing_models/*/outputs/sweep/threshold_sweep.csv"))


def main():
    if not CSVS:
        raise SystemExit("No threshold_sweep.csv files found yet.")
    df = pd.concat([pd.read_csv(c) for c in CSVS], ignore_index=True)
    df = df.drop_duplicates(subset=["model", "task", "split", "threshold"], keep="last")
    df = df.sort_values(["task", "model", "split", "threshold"])

    out_csv = os.path.join(REPO, "docs/threshold_sweep_combined.csv")
    df.to_csv(out_csv, index=False)

    # Best F1 per (model, task, split)
    valid = df[df["f1"] > 0]
    best = valid.loc[valid.groupby(["model", "task", "split"])["f1"].idxmax()]
    best = best.sort_values(["task", "model", "split"])

    ap_col = "ap40" if "ap40" in df.columns else "ap50"

    lines = ["# Score-threshold sweep (3 imgs/source, random split)\n",
             "## Best-F1 threshold per model\n",
             f"| Task | Model | Split | Best thr | Recall | Precision | {ap_col.upper()} | F1 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for _, r in best.iterrows():
        lines.append(f"| {r.task} | {r.model} | {r.split} | {r.threshold:.2f} | "
                     f"{r.recall:.3f} | {r.precision:.3f} | {r[ap_col]:.3f} | {r.f1:.3f} |")

    lines.append("\n## Full grid\n")
    for (task, model, split), g in df.groupby(["task", "model", "split"]):
        lines.append(f"\n### {task} / {model} / {split}\n")
        lines.append(f"| thr | recall | precision | {ap_col} | f1 |")
        lines.append("|---:|---:|---:|---:|---:|")
        for _, r in g.iterrows():
            lines.append(f"| {r.threshold:.2f} | {r.recall:.3f} | {r.precision:.3f} | "
                         f"{r[ap_col]:.3f} | {r.f1:.3f} |")

    md = "\n".join(lines) + "\n"
    out_md = os.path.join(REPO, "docs/threshold_sweep_summary.md")
    with open(out_md, "w") as fh:
        fh.write(md)

    print(f"Found {len(CSVS)} CSV(s); {len(df)} rows.")
    print("\n".join(lines[:4 + len(best)]))
    print(f"\nWrote {out_csv}\n      {out_md}")


if __name__ == "__main__":
    main()
