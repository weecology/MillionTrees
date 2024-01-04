import glob
import pandas as pd
from deepforest.utilities import read_file

def Ryoungseob_2023():
    xmls = glob.glob("/blue/ewhite/DeepForest/Ryoungseob_2023/train_datasets/annotations/*.xml")

    #Load and format xmls
    annotation_list = []   
    for xml_path in xmls:
        try:
            annotation = read_file(xml_path)
        except:
            continue
        annotation_list.append(annotation)
    annotations = pd.concat(annotation_list, ignore_index=True)      
    annotations["label"] = "Tree"
    annotations["source"] = "Kwon et al. 2023"  
    annotations["split"] = "train"
    
    # Train only
    annotations.to_csv("/blue/ewhite/DeepForest/Ryoungseob_2023/train_datasets/images/train.csv")