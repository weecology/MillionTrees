"""
Prepare NIBIO UAV tree damage dataset (Puliti and Astrup 2022) for MillionTrees.

Zenodo: https://zenodo.org/records/14711562
Data: YOLO-format bounding boxes (class x_center y_center width height, normalized).
All classes are combined as 'tree' for the benchmark.

Output: annotations.csv with image_path, geometry (WKT box), source, for use by
package_datasets.py (TreeBoxes).
"""

from pathlib import Path
import pandas as pd
from shapely.geometry import box
from PIL import Image

ROOT = Path("/orange/ewhite/DeepForest/Puliti_2022")
OUT_CSV = ROOT / "annotations.csv"
SOURCE_NAME = "Puliti and Astrup 2022"


def yolo_line_to_pixel_box(line: str, img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """Convert one YOLO line (class x_center y_center w h normalized) to pixel xmin, ymin, xmax, ymax."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    _class = parts[0]
    x_center = float(parts[1])
    y_center = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    xmin = (x_center - w / 2) * img_w
    ymin = (y_center - h / 2) * img_h
    xmax = (x_center + w / 2) * img_w
    ymax = (y_center + h / 2) * img_h
    # Clamp to image
    xmin = max(0, min(img_w, xmin))
    ymin = max(0, min(img_h, ymin))
    xmax = max(0, min(img_w, xmax))
    ymax = max(0, min(img_h, ymax))
    if xmin >= xmax or ymin >= ymax:
        return None
    return (xmin, ymin, xmax, ymax)


def run(root: Path | None = None, out_csv: Path | None = None) -> pd.DataFrame:
    root = root or ROOT
    out_csv = out_csv or OUT_CSV
    rows = []
    for png_path in sorted(root.glob("*.png")):
        label_path = png_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        try:
            with Image.open(png_path) as img:
                img_w, img_h = img.size
        except Exception:
            continue
        image_path = str(png_path.resolve())
        with open(label_path) as f:
            for line in f:
                if not line.strip():
                    continue
                b = yolo_line_to_pixel_box(line, img_w, img_h)
                if b is None:
                    continue
                xmin, ymin, xmax, ymax = b
                geom = box(xmin, ymin, xmax, ymax)
                rows.append({
                    "image_path": image_path,
                    "geometry": geom.wkt,
                    "source": SOURCE_NAME,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No annotations found under {root}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} annotations, {df['image_path'].nunique()} images -> {out_csv}")
    return df


def plot_sample_overlay(
    root: Path | None = None,
    df: pd.DataFrame | None = None,
    sample_image_path: str | None = None,
    savedir: Path | str | None = None,
):
    """Draw one image with overlaid boxes and save to savedir (default: ROOT on /orange).
    Source images are RGBA; we strip alpha and work in RGB so saved overlay has correct colors.
    """
    import cv2
    import numpy as np
    import shapely.wkt

    root = root or ROOT
    if df is None:
        df = pd.read_csv(OUT_CSV)
    if sample_image_path is None:
        sample_image_path = df["image_path"].iloc[0]
    if savedir is None:
        savedir = root / "sample_overlay"
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    sub = df[df["image_path"] == sample_image_path].copy()
    if sub.empty:
        raise ValueError(f"No annotations for {sample_image_path}")
    # Load as RGB (PIL handles RGBA; drop alpha so we have 3 channels)
    with Image.open(sample_image_path) as pil_img:
        rgb = np.array(pil_img.convert("RGB"))
    # cv2 uses BGR for drawing
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for _, row in sub.iterrows():
        geom = shapely.wkt.loads(row["geometry"])
        xmin, ymin, xmax, ymax = geom.bounds
        xmin, ymin = int(round(xmin)), int(round(ymin))
        xmax, ymax = int(round(xmax)), int(round(ymax))
        cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    # Save as RGB so viewers show correct colors
    out_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out_path = savedir / "Puliti_2022_sample.png"
    Image.fromarray(out_rgb).save(str(out_path))
    print(f"Sample overlay saved to {out_path}")


if __name__ == "__main__":
    df = run()
    plot_sample_overlay(df=df)
