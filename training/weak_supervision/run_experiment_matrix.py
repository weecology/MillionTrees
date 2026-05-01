"""Run scratch vs box-pretrained polygon experiments (subset and optional full track)."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def run_cmd(command, cwd):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {
        "command": " ".join(command),
        "returncode": result.returncode,
        "stdout": result.stdout[-6000:],
        "stderr": result.stderr[-6000:],
    }


def main():
    parser = argparse.ArgumentParser(description="Run weak supervision experiment matrix.")
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--root-dir", type=str, default="data-mini")
    parser.add_argument("--split-scheme", type=str, default="random")
    parser.add_argument("--output-dir", type=str, default="training/weak_supervision/outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-mini", action="store_true")
    parser.add_argument("--download-subset", action="store_true")
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--full-root-dir", type=str, default=None)
    parser.add_argument("--download-full", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    python_cmd = shlex.split(args.python)

    tracks = [
        {
            "name": "subset",
            "root_dir": args.root_dir,
            "mini": args.subset_mini,
            "enabled": True,
            "download": args.download_subset,
        },
        {
            "name": "full",
            "root_dir": args.full_root_dir or args.root_dir,
            "mini": False,
            "enabled": args.run_full,
            "download": args.download_full,
        },
    ]

    report = {"split_scheme": args.split_scheme, "tracks": []}
    for track in tracks:
        if not track["enabled"]:
            continue
        track_dir = out_dir / track["name"]
        track_dir.mkdir(parents=True, exist_ok=True)

        pretrain_dir = track_dir / "box_pretrain"
        scratch_dir = track_dir / "polygon_scratch"
        boxinit_dir = track_dir / "polygon_box_pretrained"
        pretrain_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        boxinit_dir.mkdir(parents=True, exist_ok=True)

        mini_flag = ["--mini"] if track["mini"] else []
        download_flag = ["--download"] if track["download"] else []

        pretrain_cmd = [
            *python_cmd,
            "training/boxes/pretrain_backbone_for_polygons.py",
            "--root-dir",
            track["root_dir"],
            "--split-scheme",
            args.split_scheme,
            "--include-unsupervised",
            "--max-epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--limit-train-batches",
            "2",
            "--limit-val-batches",
            "2",
            "--eval-max-batches",
            "2",
            "--data-scope",
            track["name"],
            "--seed",
            str(args.seed),
            "--output-dir",
            str(pretrain_dir),
            *mini_flag,
            *download_flag,
        ]
        pretrain_res = run_cmd(pretrain_cmd, cwd=str(repo_root))

        box_backbone = pretrain_dir / f"box_backbone_{args.split_scheme}.pt"
        scratch_cmd = [
            *python_cmd,
            "training/polygons/train_polygons.py",
            "--root-dir",
            track["root_dir"],
            "--split-scheme",
            args.split_scheme,
            "--max-epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--gpus",
            "1",
            "--limit-train-batches",
            "2",
            "--limit-val-batches",
            "2",
            "--seed",
            str(args.seed),
            "--init-mode",
            "scratch",
            "--data-scope",
            track["name"],
            "--output-dir",
            str(scratch_dir),
            *mini_flag,
            *download_flag,
        ]
        scratch_res = run_cmd(scratch_cmd, cwd=str(repo_root))

        boxinit_cmd = [
            *python_cmd,
            "training/polygons/train_polygons.py",
            "--root-dir",
            track["root_dir"],
            "--split-scheme",
            args.split_scheme,
            "--max-epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--gpus",
            "1",
            "--limit-train-batches",
            "2",
            "--limit-val-batches",
            "2",
            "--seed",
            str(args.seed),
            "--init-mode",
            "box_pretrained",
            "--box-backbone-checkpoint",
            str(box_backbone),
            "--data-scope",
            track["name"],
            "--output-dir",
            str(boxinit_dir),
            *mini_flag,
            *download_flag,
        ]
        boxinit_res = run_cmd(boxinit_cmd, cwd=str(repo_root))

        track_record = {
            "name": track["name"],
            "root_dir": track["root_dir"],
            "mini": track["mini"],
            "download": track["download"],
            "pretrain": pretrain_res,
            "polygon_scratch": scratch_res,
            "polygon_box_pretrained": boxinit_res,
        }
        report["tracks"].append(track_record)

    report_path = out_dir / "experiment_matrix_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote experiment matrix report to {report_path}")


if __name__ == "__main__":
    main()
