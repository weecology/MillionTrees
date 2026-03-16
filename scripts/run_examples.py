"""Run existing-model evaluations and optionally generate the benchmark table.

This script orchestrates evaluation of pretrained models (DeepForest, SAM3) on the
MillionTrees test set. Each model is a self-contained subpackage under existing_models/
with its own uv environment to avoid dependency conflicts.

For fine-tuned model training + full benchmark, use slurm/run_benchmark.sbatch instead.
"""

import argparse
import os
import subprocess
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd: List[str], cwd: str) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def eval_deepforest(root_dir: str, split: str, mini: bool, max_batches: int | None,
                    output_dir: str | None) -> None:
    cwd = os.path.join(ROOT, "existing_models", "deepforest")
    common = ["--root-dir", root_dir, "--split-scheme", split]
    if mini:
        common += ["--mini"]
    if max_batches:
        common += ["--max-batches", str(max_batches)]
    if output_dir:
        common += ["--output-dir", output_dir]
    for script in ["eval_boxes.py", "eval_points.py", "eval_polygons.py"]:
        run(["uv", "run", "python", script] + common, cwd=cwd)


def eval_sam3(root_dir: str, split: str, mini: bool, max_batches: int | None,
              device: str, output_dir: str | None) -> None:
    cwd = os.path.join(ROOT, "existing_models", "sam3")
    common = ["--root-dir", root_dir, "--split-scheme", split, "--device", device]
    if mini:
        common += ["--mini"]
    if max_batches:
        common += ["--max-batches", str(max_batches)]
    if output_dir:
        common += ["--output-dir", output_dir]
    for script in ["eval_boxes.py", "eval_points.py", "eval_polygons.py"]:
        run(["uv", "run", "python", script] + common, cwd=cwd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run existing-model evaluations on MillionTrees.")
    ap.add_argument("--root-dir", type=str,
                    default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    ap.add_argument("--split-scheme", type=str, default="random",
                    choices=["random", "zeroshot", "crossgeometry"])
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=["deepforest", "sam3"],
                    choices=["deepforest", "sam3"], help="Which models to evaluate")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Save results here (default: existing_models/{model}/outputs/{split})")
    ap.add_argument("--make-table", action="store_true",
                    help="Run make_benchmark_table.py after evaluations")
    args = ap.parse_args()

    out_base = args.output_dir

    if "deepforest" in args.models:
        out = out_base or os.path.join(ROOT, "existing_models", "deepforest", "outputs",
                                       args.split_scheme)
        eval_deepforest(args.root_dir, args.split_scheme, args.mini, args.max_batches, out)

    if "sam3" in args.models:
        out = out_base or os.path.join(ROOT, "existing_models", "sam3", "outputs",
                                       args.split_scheme)
        eval_sam3(args.root_dir, args.split_scheme, args.mini, args.max_batches, args.device, out)

    if args.make_table:
        run(["uv", "run", "python", "scripts/make_benchmark_table.py",
             "--splits", args.split_scheme], cwd=ROOT)


if __name__ == "__main__":
    main()
