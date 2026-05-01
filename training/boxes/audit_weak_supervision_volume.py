"""Audit weak-supervision annotation volume for TreeBoxes (CSV-level counts)."""

import argparse
import fnmatch
import json
from pathlib import Path

import pandas as pd

from milliontrees import get_dataset


def _to_patterns(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _match_any(text, patterns):
    if not patterns:
        return False
    text = text.lower()
    return any(fnmatch.fnmatch(text, pattern.lower()) for pattern in patterns)


def main():
    parser = argparse.ArgumentParser(description="Count weak-supervision annotations for TreeBoxes.")
    parser.add_argument("--root-dir", type=str, default="data")
    parser.add_argument("--split-scheme", type=str, default="random")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--include-unsupervised", action="store_true")
    parser.add_argument(
        "--weak-patterns",
        nargs="+",
        default=["*unsupervised*", "*weak supervised*", "*neon*"],
        help="Source wildcard patterns considered weak supervision.",
    )
    parser.add_argument("--include-sources", nargs="+", default=None)
    parser.add_argument("--exclude-sources", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="training/boxes/audit_outputs")
    parser.add_argument(
        "--data-scope",
        type=str,
        default="subset",
        choices=["subset", "full"],
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / f"weak_volume_summary_{args.split_scheme}_{args.data_scope}.json"

    try:
        dataset = get_dataset(
            "TreeBoxes",
            root_dir=args.root_dir,
            split_scheme=args.split_scheme,
            mini=args.mini,
            download=args.download,
            include_unsupervised=args.include_unsupervised,
            include_sources=_to_patterns(args.include_sources),
            exclude_sources=_to_patterns(args.exclude_sources),
        )
    except Exception as exc:
        failed = {
            "task": "TreeBoxes",
            "split": args.split_scheme,
            "data_scope": args.data_scope,
            "include_unsupervised": args.include_unsupervised,
            "status": "failed",
            "error": str(exc),
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        print(json.dumps(failed, indent=2))
        return

    split_csv = Path(dataset._data_dir) / f"{args.split_scheme}.csv"
    df_raw = pd.read_csv(split_csv, low_memory=False)
    weak_mask_all = df_raw["source"].astype(str).apply(
        lambda s: _match_any(s, args.weak_patterns)
    )
    df_all_weak = df_raw[weak_mask_all]

    include_patterns = _to_patterns(args.include_sources)
    exclude_patterns = _to_patterns(args.exclude_sources)
    if exclude_patterns is None:
        exclude_patterns = [] if args.include_unsupervised else ["*unsupervised*"]

    source_series = df_raw["source"].astype(str)
    selected_mask = pd.Series(True, index=df_raw.index)
    if include_patterns:
        selected_mask &= source_series.apply(lambda s: _match_any(s, include_patterns))
    if exclude_patterns:
        selected_mask &= ~source_series.apply(lambda s: _match_any(s, exclude_patterns))
    df_selected = df_raw[selected_mask]

    weak_mask_selected = df_selected["source"].astype(str).apply(
        lambda s: _match_any(s, args.weak_patterns)
    )
    df_selected_weak = df_selected[weak_mask_selected]

    grouped = (
        df_selected_weak.groupby(["source", "split"], dropna=False)
        .size()
        .reset_index(name="n_annotations")
        .sort_values("n_annotations", ascending=False)
    )

    counts_csv = out_dir / f"weak_source_counts_{args.split_scheme}_{args.data_scope}.csv"
    grouped.to_csv(counts_csv, index=False)

    summary = {
        "task": "TreeBoxes",
        "split": args.split_scheme,
        "data_scope": args.data_scope,
        "include_unsupervised": args.include_unsupervised,
        "weak_patterns": args.weak_patterns,
        "total_annotations_raw": int(len(df_raw)),
        "total_annotations_selected": int(len(df_selected)),
        "weak_annotations_raw": int(len(df_all_weak)),
        "weak_annotations_selected": int(len(df_selected_weak)),
        "weak_sources_selected": int(df_selected_weak["source"].nunique()),
        "raw_csv": str(split_csv.resolve()),
        "weak_source_counts_csv": str(counts_csv.resolve()),
        "status": "ok",
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
