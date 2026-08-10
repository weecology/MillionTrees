Add a new source dataset to MillionTrees. The user will describe the dataset (name, location of raw data, geometry type, citation). Follow every step below in order.

## Step 1 — Determine geometry type

Ask the user (or infer from context) whether the annotations are:
- **Points** → `[TreePoints]` section
- **Boxes** (bounding rectangles) → `[TreeBoxes]` section
- **Polygons** (crown outlines) → `[TreePolygons]` section

This controls the section in `annotation_csvs.cfg`, the output CSV filename prefix (`TreePoints_`, `TreeBoxes_`, or `TreePolygons_`), and how the geometry column is stored.

## Step 2 — Write the data_prep script

Create `data_prep/<DatasetName>.py` following the pattern of existing scripts (e.g. `Ventura2022.py`, `process_ofo_field.py`).

**Required output schema** (from `docs/contributing.md`):
```
image_path  — absolute path to the image file on disk
source      — abbreviated citation, e.g. "Smith et al. 2023"
geometry    — WKT string in *image* pixel coordinates (origin 0,0 = top-left).
              NOT the geographic CRS. Use deepforest.utilities.read_file() to
              convert from geographic coords when needed.
existing_split — (optional) "train" or "test" if the source has a canonical split
```

**Key rules:**
- `image_path` must be a **full absolute path**, never a relative path or basename.
- `source` must be set to a consistent citation string used in `docs/datasets.md`.
- Use `deepforest.utilities.read_file()` to convert geo-referenced shapefiles/GeoDataFrames to image-pixel coordinates. Pass `root_dir=` pointing to the directory containing the .tif.
- Do not include species, DBH, or any field metadata columns — strip them.
- Save the final CSV alongside the images, e.g. `<data_root>/annotations.csv`.

**Tiling large .tif files:**
If source images are large orthomosaics (typically > 2000 px in either dimension), tile them with `deepforest.preprocess.split_raster` before writing the CSV:

```python
from deepforest.preprocess import split_raster
from deepforest.utilities import read_file
import geopandas as gpd, pandas as pd, os

gdf = read_file(annotations_df, root_dir=img_dir)  # converts geo -> pixel coords
tiles = split_raster(
    gdf,
    path_to_raster=tif_path,
    root_dir=img_dir,
    save_dir=images_dir,
    patch_size=800,
    patch_overlap=0,
    allow_empty=False,
)
tiles = tiles.rename(columns={"image_path": "filename"})
tiles["image_path"] = tiles["filename"].apply(lambda f: os.path.join(images_dir, f))
tiles["source"] = "Author et al. YEAR"
tiles.to_csv(os.path.join(output_dir, "annotations.csv"), index=False)
```

## Step 3 — Write a verification overlay image

After generating the CSV, produce at least one overlay image that draws the annotations on top of the corresponding tile/image and saves it **in the same directory as the source images**, so visual alignment can be verified without running the full pipeline:

```python
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd, os, shapely.wkt

def write_overlay(csv_path: str, images_dir: str, max_samples: int = 4):
    df = pd.read_csv(csv_path)
    for img_path in sorted(df["image_path"].unique())[:max_samples]:
        sub = df[df["image_path"] == img_path]
        img = np.array(Image.open(img_path).convert("RGB"))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        for _, row in sub.iterrows():
            geom = shapely.wkt.loads(row["geometry"])
            gt = geom.geom_type
            if gt == "Point":
                ax.plot(geom.x, geom.y, "c+", markersize=10, markeredgewidth=2)
            elif gt == "Polygon":
                xs, ys = geom.exterior.xy
                ax.plot(xs, ys, "c-", linewidth=1.5)
            else:  # box / LineString used as box representation
                b = geom.bounds
                rect = patches.Rectangle(
                    (b[0], b[1]), b[2]-b[0], b[3]-b[1],
                    linewidth=1.5, edgecolor="cyan", facecolor="none"
                )
                ax.add_patch(rect)
        ax.set_title(f"{os.path.basename(img_path)} — {len(sub)} annotations")
        ax.axis("off")
        out = os.path.join(images_dir, "overlay_" + os.path.splitext(os.path.basename(img_path))[0] + ".png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Overlay saved: {out}")
```

Call `write_overlay(csv_path, images_dir)` at the end of the prep script.

## Step 4 — Add the CSV to `annotation_csvs.cfg`

Open `data_prep/annotation_csvs.cfg` and add the **full absolute path** to the new CSV in the correct geometry section:

```
[TreePoints]          ← or TreeBoxes / TreePolygons
...existing entries...
/orange/ewhite/DeepForest/<Dataset>/annotations.csv   ← new entry
```

Never add a commented-out line unless the dataset is intentionally excluded.

## Step 5 — Add documentation in `docs/datasets.md`

Add a new subsection under the appropriate geometry heading (`# Points`, `# Boxes`, or `# Polygons`). Pattern:

```markdown
## <Author(s)> <Year>

### Source Name: "<source string used in annotations>"

![sample_image](public/<SourceName>.png)

**Citation:** <full citation>

**Location:** <geographic region>

<1-2 sentences describing the dataset: what kind of trees, how annotations were made, sensor type.>
```

The `public/<SourceName>.png` image is auto-generated by `package_datasets.py` → `create_mini_datasets()`; it will appear after packaging. Add a placeholder reference now so docs are complete.

## Step 6 — Verify the version in Tree dataset files matches `package_datasets.py`

Check the current version in `data_prep/package_datasets.py`:
```bash
grep "version\s*=" data_prep/package_datasets.py
```

Then check `src/milliontrees/datasets/Tree{Points,Boxes,Polygons}.py` to confirm the latest version key in `_versions_dict` matches. If `package_datasets.py` version is newer (e.g. `v0.14`) but the dataset file only goes up to `0.12`, a new entry must be added after packaging with the real download URL and compressed size. Flag this to the user — the URLs can only be filled in after the package has been uploaded to `data.rc.ufl.edu`.

## Step 7 — Run packaging

Submit the packaging job from the repo root. The canonical command is:
```bash
cd /blue/ewhite/b.weinstein/src/MillionTrees
uv run python data_prep/package_datasets.py
```

For long runs on HiPerGator, submit via SLURM. Look at `slurm/submit_all.sh` or create a per-dataset submit script modeled on `slurm/submit_ofo_field.sh`:
```bash
sbatch slurm/submit_<datasetname>.sh
```

After packaging completes, confirm:
- `TreePoints_<version>/random.csv` (or Boxes/Polygons) has rows from the new source.
- `docs/public/<SourceName>.png` was generated.
- Images were copied into `Tree{Points,Boxes,Polygons}_<version>/images/`.

## Step 8 — Update download URLs (after upload)

Once the new package zip has been uploaded to `https://data.rc.ufl.edu/pub/ewhite/MillionTrees/`, update `_versions_dict` in all three dataset files to add the new version entry:

```python
# src/milliontrees/datasets/TreePoints.py  (and TreeBoxes.py, TreePolygons.py)
_versions_dict = {
    ...
    "0.14": {
        "download_url": "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_v0.14.zip",
        "supervised_download_url": "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_supervised_v0.14.zip",
        "compressed_size": <size in bytes>,
    }
}
```

Get the compressed size with:
```bash
stat --printf="%s\n" /path/to/TreePoints_v0.14.zip
```

## Checklist

Before declaring the dataset added, confirm:
- [ ] `data_prep/<DatasetName>.py` exists and runs without errors
- [ ] CSV has columns: `image_path` (absolute), `source`, `geometry` (image-pixel WKT)
- [ ] `annotation_csvs.cfg` has the new CSV path in the right section
- [ ] At least one overlay PNG exists in the images dir for visual QC
- [ ] `docs/datasets.md` has a new section for this source
- [ ] `package_datasets.py` version matches or exceeds the latest `_versions_dict` key
- [ ] Packaging has been run and new source appears in the output CSV
- [ ] Download URLs updated in `Tree{Points,Boxes,Polygons}.py` after upload
