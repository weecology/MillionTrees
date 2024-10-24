import os
import pandas as pd
from deepforest.utilities import read_file
import numpy as np
import xmltodict


def xml_to_annotations(xml_path, image_dir):
    """
    Load annotations from xml format (e.g. RectLabel editor) and convert
    them into retinanet annotations format.
    Args:
        xml_path (str): Path to the annotations xml, formatted by RectLabel
        image_dir: Directory to search for images
    Returns:
        Annotations (pandas dataframe): in the
            format -> path-to-image.png,x1,y1,x2,y2,class_name
    """
    # parse
    with open(xml_path) as fd:
        doc = xmltodict.parse(fd.read())

    # grab xml objects
    try:
        tile_xml = doc["annotation"]["object"]
    except Exception as e:
        raise Exception("error {} for path {} with doc annotation{}".format(
            e, xml_path, doc["annotation"]))

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    label = []

    if isinstance(tile_xml, list):
        # Construct frame if multiple trees
        for tree in tile_xml:
            xmin.append(tree["bndbox"]["xmin"])
            xmax.append(tree["bndbox"]["xmax"])
            ymin.append(tree["bndbox"]["ymin"])
            ymax.append(tree["bndbox"]["ymax"])
    else:
        xmin.append(tile_xml["bndbox"]["xmin"])
        xmax.append(tile_xml["bndbox"]["xmax"])
        ymin.append(tile_xml["bndbox"]["ymin"])
        ymax.append(tile_xml["bndbox"]["ymax"])

    rgb_name = os.path.basename(doc["annotation"]["filename"])
    
    if os.path.exists("{}/{}".format(image_dir,rgb_name)):
        # set dtypes, check for floats and round
        xmin = [int(np.round(float(x))) for x in xmin]
        xmax = [int(np.round(float(x)))for x in xmax]
        ymin = [int(np.round(float(x))) for x in ymin]
        ymax = [int(np.round(float(x))) for x in ymax]
    
        annotations = pd.DataFrame({
            "image_path": rgb_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })
        return (annotations)
    else:
        raise IOError("{} doesn't exist".format(rgb_name))
    
def recursive_search_and_process(root_dir):
    xml_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(subdir, file))
    
    dataframes = []
    for xml_file in xml_files:
        try:
            df = xml_to_annotations(xml_file, root_dir + "/images")
        except Exception as e:
            print("Error processing file: {}. Error: {}".format(xml_file, e))
            continue
        df['image_path'] = os.path.join(root_dir + "/images", os.path.splitext(os.path.basename(xml_file))[0] + '.JPG')
        df = read_file(df, root_dir + "/images")
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['label'] = 'Tree'
    combined_df['source'] = 'Radogoshi et al. 2021'
    
    combined_df.to_csv(os.path.join(root_dir, 'annotations.csv'), index=False)

root_directory = "/orange/ewhite/DeepForest/Radogoshi_Sweden"
recursive_search_and_process(root_directory)