import pandas as pd
from deepforest.utilities import read_file

df = pd.read_csv("/orange/ewhite/DeepForest/Zenodo_15155081/jelled_annotations_finetuning_DF/annotations.csv")
df["source"] = "Dumortier et al. 2025"
gdf = read_file(df)
gdf["image_path"] = gdf.image_path.apply(lambda x: "/orange/ewhite/DeepForest/Zenodo_15155081/jelled_annotations_finetuning_DF/images/{}".format(x))
gdf.to_csv("/orange/ewhite/DeepForest/Zenodo_15155081/parsed_annotations.csv", index=False)

