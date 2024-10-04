import glob
import os
import shutil
import pandas as pd
from deepforest.utilities import read_file

BENCHMARK_PATH = "/orange/idtrees-collab/NeonTreeEvaluation/"
tifs = glob.glob(BENCHMARK_PATH + "evaluation/RGB/*.tif")
xmls = [os.path.splitext(os.path.basename(x))[0] for x in tifs] 
xmls = [os.path.join(BENCHMARK_PATH, "annotations", x) + ".xml" for x in xmls] 

#Load and format xmls, not every RGB image has an annotation
annotation_list = []   
for xml_path in xmls:
    try:
        annotation = read_file(xml_path)
    except:
        continue
    annotation_list.append(annotation)
benchmark_annotations = pd.concat(annotation_list, ignore_index=True)

benchmark_annotations["source"] = "NEON_benchmark"
benchmark_annotations["label"] = "Tree"

for image_path in benchmark_annotations.image_path.unique():
    dst = os.path.join(BENCHMARK_PATH, "evaluation/RGB/", image_path)
    shutil.copy(dst, "/orange/ewhite/DeepForest/NEON_benchmark/images/")

benchmark_annotations["image_path"] = benchmark_annotations.image_path.apply(lambda x: os.path.join("/orange/ewhite/DeepForest/NEON_benchmark/images/", x))
benchmark_annotations.to_csv("/orange/ewhite/DeepForest/NEON_benchmark/NeonTreeEvaluation_annotations.csv")