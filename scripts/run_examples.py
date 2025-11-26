import argparse
import os
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any

ROOT = os.path.dirname(os.path.dirname(__file__))


@dataclass
class ExampleSpec:
    name: str
    task: str
    script: str
    args: List[str]


def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.stdout


def build_examples(root_dir: str, split_scheme: str, mini: bool, device: str, max_batches: int | None) -> List[ExampleSpec]:
    common = []
    if mini:
        common += ["--mini"]
    if split_scheme:
        common += ["--split-scheme", split_scheme]
    ex: List[ExampleSpec] = [
        ExampleSpec(
            name="sam3_points.py",
            task="TreePoints",
            script="docs/examples/sam3_points.py",
            args=["--backend", "native", "--root-dir", root_dir, "--device", device] + (["--max-batches", str(max_batches)] if max_batches else []) + common,
        ),
        ExampleSpec(
            name="sam3_boxes.py",
            task="TreeBoxes",
            script="docs/examples/sam3_boxes.py",
            args=["--backend", "native", "--root-dir", root_dir, "--device", device] + (["--max-batches", str(max_batches)] if max_batches else []) + common,
        ),
        ExampleSpec(
            name="baseline_points.py",
            task="TreePoints",
            script="docs/examples/baseline_points.py",
            args=["--root-dir", root_dir] + (["--max-batches", str(max_batches)] if max_batches else []) + common,
        ),
        ExampleSpec(
            name="baseline_boxes.py",
            task="TreeBoxes",
            script="docs/examples/baseline_boxes.py",
            args=["--root-dir", root_dir] + (["--max-batches", str(max_batches)] if max_batches else []) + common,
        ),
        # Polygons (enable when dataset root has 'polygon' column)
        # ExampleSpec(
        #     name="sam3_polygons.py",
        #     task="TreePolygons",
        #     script="docs/examples/sam3_polygons.py",
        #     args=["--backend", "native", "--root-dir", root_dir, "--device", device] + (["--max-batches", str(max_batches)] if max_batches else []) + common,
        # ),
        # ExampleSpec(
        #     name="baseline_polygons.py",
        #     task="TreePolygons",
        #     script="docs/examples/baseline_polygons.py",
        #     args=["--root-dir", root_dir] + (["--max-batches", str(max_batches)] if max_batches else []) + common,
        # ),
    ]
    return ex


def parse_metrics(output: str) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    for line in output.splitlines():
        s = line.strip().lower()
        if s.startswith("average keypointaccuracy:"):
            m["KeypointAccuracy"] = float(s.split(":")[1].strip())
        if s.startswith("average counting_mae:"):
            m["CountingMAE"] = float(s.split(":")[1].strip())
        if s.startswith("average detection_accuracy:"):
            m["DetectionAccuracy"] = float(s.split(":")[1].strip())
        if s.startswith("average detection_recall:"):
            m["DetectionRecall"] = float(s.split(":")[1].strip())
    return m


def update_leaderboard(entries: List[Dict[str, Any]]) -> None:
    leaderboard_path = os.path.join(ROOT, "docs", "leaderboard.md")
    with open(leaderboard_path, "r", encoding="utf-8") as f:
        md = f.read()
    header = "## Generated results"
    start = md.find(header)
    table_header = (
        "| Model | Task | Split | Dataset | Size | Script |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for e in entries:
        rows.append(
            f"| {e['model']} | {e['task']} | {e['split']} | {e['dataset']} | {e['size']} | `{e['script']}` |"
        )
    table = header + "\n\n" + table_header + "\n".join(rows) + "\n"
    if start == -1:
        md = md.rstrip() + "\n\n" + table
    else:
        # replace existing block
        md = md[:start] + table
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        f.write(md)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run MillionTrees examples and update leaderboard.")
    ap.add_argument("--root-dir", type=str, default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    ap.add_argument("--split-scheme", type=str, default="random", choices=["random", "zeroshot", "crossgeometry"])
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()

    specs = build_examples(args.root_dir, args.split_scheme, args.mini, args.device, args.max_batches)
    results_summary: List[Dict[str, Any]] = []
    for spec in specs:
        cmd = ["uv", "run", "python", spec.script] + spec.args
        out = run_cmd(cmd)
        metrics = parse_metrics(out)
        entry = {
            "model": spec.name,
            "task": spec.task,
            "split": args.split_scheme,
            "dataset": spec.task,
            "size": "mini" if args.mini else "full",
            "script": " ".join(cmd),
            "metrics": metrics,
        }
        results_summary.append(entry)

    update_leaderboard(results_summary)


if __name__ == "__main__":
    main()



