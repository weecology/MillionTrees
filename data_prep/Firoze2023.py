import os
import json
import pandas as pd

def extract_bounding_boxes_from_labelme(folder_path):
    data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            json_path = os.path.join(folder_path, filename)
            with open(json_path, 'r') as file:
                labelme_data = json.load(file)
                image_path = os.path.join(folder_path, labelme_data['imagePath'])
                
                polygons = []
                for shape in labelme_data['shapes']:
                    points = shape['points']
                    # Add the first to the end to close the polygon
                    points.append(points[0])
                    wkt_polygon = 'POLYGON(({}))'.format(', '.join(['{} {}'.format(p[0], p[1]) for p in points]))
                    polygons.append([image_path, wkt_polygon])
                
                # Create dataframe with image path
                df = pd.DataFrame(polygons, columns=['image_path', 'geometry'])
                data.append(df)

    annotations = pd.concat(data)
    
    return annotations

# Example usage
folder_path = '/orange/ewhite/DeepForest/Firoze2023/annotated_forest_dataset/annotated_real_forest'
df = extract_bounding_boxes_from_labelme(folder_path)
df["source"] = "Firoze et al. 2023"
df["label"] = "Tree"
print("There are {} annotations in {} images".format(df.shape[0], len(df.image_path.unique())))
df.to_csv("/orange/ewhite/DeepForest/Firoze2023/annotations.csv")

