import os
import pandas as pd
from deepforest.preprocess import split_raster

IMAGES_DIR = "/orange/ewhite/DeepForest/ReForestTree/images"
TILES_DIR = "/orange/ewhite/DeepForest/ReForestTree/tiles"
PATCH_SIZE = 1500


def ReForestTree():
    """This dataset used deepforest to generate predictions which were cleaned, no test data can be used"""
    os.makedirs(TILES_DIR, exist_ok=True)

    annotations = pd.read_csv("/orange/ewhite/DeepForest/ReForestTree/mapping/final_dataset.csv")
    annotations["image_path"] = IMAGES_DIR + "/" + annotations["img_path"]
    annotations["source"] = "Reiersen et al. 2022"
    annotations["label"] = "Tree"
    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))

    annotations["xmin"] = annotations["xmin"].astype(int)
    annotations["ymin"] = annotations["ymin"].astype(int)
    annotations["xmax"] = annotations["xmax"].astype(int)
    annotations["ymax"] = annotations["ymax"].astype(int)

    annotations = annotations[["image_path", "xmin", "ymin", "xmax", "ymax", "label", "source"]]

    tiled = []
    for image_path in annotations["image_path"].unique():
        image_annotations = annotations[annotations["image_path"] == image_path].copy()
        image_annotations["image_path"] = os.path.basename(image_path)
        try:
            crop = split_raster(
                image_annotations,
                path_to_raster=image_path,
                root_dir=IMAGES_DIR,
                patch_size=PATCH_SIZE,
                patch_overlap=0,
                allow_empty=False,
                save_dir=TILES_DIR,
            )
            tiled.append(crop)
        except Exception as e:
            print(f"Skipping {image_path}: {e}")

    combined = pd.concat(tiled, ignore_index=True)
    combined["image_path"] = combined["image_path"].apply(
        lambda x: os.path.join(TILES_DIR, os.path.basename(x))
    )
    combined["source"] = "Reiersen et al. 2022"
    combined["label"] = "Tree"
    combined.to_csv(os.path.join(TILES_DIR, "train.csv"), index=False)
    print(f"Saved {len(combined)} annotations across {combined['image_path'].nunique()} tiles")
    return combined


ReForestTree()
