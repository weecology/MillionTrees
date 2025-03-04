import paramiko
import os
import datetime
import pandas as pd
from label_studio_sdk import Client
import glob
import shutil
from PIL import Image

def upload_to_label_studio(images, sftp_client, url, project_name, images_to_annotate_dir, folder_name, preannotations, dataset_type, batch_size=10):
    """
    Upload images to Label Studio and import image tasks.

    Args:
        images (list): List of image paths to upload, full paths
        url (str): The URL of the Label Studio server.
        sftp_client (paramiko.SFTPClient): The SFTP client for uploading images.
        project_name (str): The name of the Label Studio project.
        images_to_annotate_dir (str): The path to the directory of images to annotate.
        folder_name (str): The name of the folder to upload images to.
        preannotations (list): List of preannotations for the images.
        dataset_type (str): Type of dataset being uploaded.
        batch_size (int, optional): Number of images per batch. Defaults to 10.

    Returns:
        None
    """
    label_studio_project = connect_to_label_studio(url=url, project_name=project_name)
    
    # Upload images in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        upload_images(sftp_client=sftp_client, images=batch_images, folder_name=folder_name)
    
    import_image_tasks(
        label_studio_project=label_studio_project,
        image_names=images,
        local_image_dir=images_to_annotate_dir,
        predictions=preannotations,
        batch_size=batch_size,
        dataset_type=dataset_type
    )

def check_for_new_annotations(sftp_client, url, project_name, csv_dir, images_to_annotate_dir, annotated_images_dir, folder_name, dataset_type):
    """
    Check for new annotations from Label Studio, move annotated images, and gather new images to annotate.

    Args:
        sftp_client (paramiko.SFTPClient): The SFTP client for downloading images.
        url (str): The URL of the Label Studio server.
        project_name (str): The name of the Label Studio project.
        csv_dir (str): The path to the folder containing CSV files.
        images_to_annotate_dir (str): The path to the directory of images to annotate.
        annotated_images_dir (str): The path to the directory of annotated images.
        folder_name (str): The name of the folder to upload images to.
        dataset_type (str): The type of dataset being annotated.

    Returns:
        DataFrame: A DataFrame containing the gathered annotations.
    """
    label_studio_project = connect_to_label_studio(url=url, project_name=project_name)
    new_annotations = download_completed_tasks(
        label_studio_project=label_studio_project, 
        csv_dir=csv_dir,
        dataset_type=dataset_type
    )
   
    if new_annotations is not None:
        move_images(src_dir=images_to_annotate_dir, dst_dir=annotated_images_dir, annotations=new_annotations)
        
        # Download any missing images
        for image in new_annotations["image_path"].unique():
            if not os.path.exists(os.path.join(annotated_images_dir, image)):
                download_images(sftp_client=sftp_client, image_names=[image], 
                             folder_name=folder_name, local_image_dir=annotated_images_dir)
        
        delete_completed_tasks(label_studio_project=label_studio_project)
    
    return new_annotations

def label_studio_format(local_image_dir, preannotations, dataset_type):
    """Convert annotations to Label Studio format based on dataset type"""
    img = Image.open(os.path.join(local_image_dir, os.path.basename(preannotations['image_path'].iloc[0])))
    original_width, original_height = img.size
    
    results = []
    if dataset_type == 'TreePoints':
        for _, row in preannotations.iterrows():
            result = {
                "value": {
                    "x": float(row['x']/original_width*100),  # Ensure float
                    "y": float(row['y']/original_height*100),  # Ensure float
                    "width": 1.0,  # Add fixed width for visibility
                    "keypointlabels": ["tree"]  # Use fixed label
                },
                "score": 1.0,
                "to_name": "image",
                "type": "keypointlabels",
                "from_name": "label",
                "original_width": original_width,
                "original_height": original_height
            }
            results.append(result)
    
    elif dataset_type == 'TreeBoxes':
        for _, row in preannotations.iterrows():
            result = {
                "value": {
                    "x": float(row['xmin']/original_width*100),  # Ensure float
                    "y": float(row['ymin']/original_height*100),  # Ensure float
                    "width": float((row['xmax'] - row['xmin'])/original_width*100),
                    "height": float((row['ymax'] - row['ymin'])/original_height*100),
                    "rotation": 0,
                    "rectanglelabels": ["tree"]  # Use fixed label
                },
                "score": 1.0,
                "to_name": "image",
                "type": "rectanglelabels",
                "from_name": "label",
                "original_width": original_width,
                "original_height": original_height
            }
            results.append(result)
    
    return {"result": results}

# check_if_complete label studio images are done
def check_if_complete(annotations):
    """Check if any new images have been labeled.
    
    Returns:
        bool: True if new images have been labeled, False otherwise.
    """

    if annotations.shape[0] > 0:
        return True
    else:
        return False

# con for a json that looks like 
#{'id': 539, 'created_username': ' vonsteiny@gmail.com, 10', 'created_ago': '0\xa0minutes', 'task': {'id': 1962, 'data': {...}, 'meta': {}, 'created_at': '2023-01-18T20:58:48.250374Z', 'updated_at': '2023-01-18T20:58:48.250387Z', 'is_labeled': True, 'overlap': 1, 'inner_id': 381, 'total_annotations': 1, ...}, 'completed_by': {'id': 10, 'first_name': '', 'last_name': '', 'email': 'vonsteiny@gmail.com'}, 'result': [], 'was_cancelled': False, 'ground_truth': False, 'created_at': '2023-01-30T21:43:35.447447Z', 'updated_at': '2023-01-30T21:43:35.447460Z', 'lead_time': 29.346, 'parent_prediction': None, 'parent_annotation': None}
    
def convert_json_to_dataframe(x, dataset_type):
    """Convert Label Studio JSON annotations to pandas DataFrame"""
    results = []
    for annotation in x:
        if dataset_type == 'TreePoints':
            # KeyPoint format
            x = annotation["value"]["x"]/100 * annotation["original_width"]
            y = annotation["value"]["y"]/100 * annotation["original_height"]
            label = annotation["value"]["keypointlabels"][0]
            result = {
                "x": x,
                "y": y,
                "label": label,
            }
        elif dataset_type == 'TreeBoxes':
            # Rectangle format
            xmin = annotation["value"]["x"]/100 * annotation["original_width"]
            ymin = annotation["value"]["y"]/100 * annotation["original_height"]
            width = annotation["value"]["width"]/100 * annotation["original_width"]
            height = annotation["value"]["height"]/100 * annotation["original_height"]
            label = annotation["value"]["rectanglelabels"][0]
            result = {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmin + width,
                "ymax": ymin + height,
                "label": label,
            }
        elif dataset_type == 'TreePolygons':
            # Polygon format - not implemented yet
            continue
            
        results.append(result)
    
    return pd.DataFrame(results)

# Move images from images_to_annotation to images_annotated 
def move_images(annotations, src_dir, dst_dir):
    """Move images from the images_to_annotate folder to the images_annotated folder.
    Args:
        annotations (list): A list of annotations.
    
    Returns:
        None
    """
    images = annotations.image_path.unique()
    for image in images:
        src = os.path.join(src_dir, os.path.basename(image))
        dst = os.path.join(dst_dir, os.path.basename(image))
                           
        try:
            shutil.move(src, dst)
        except FileNotFoundError:
            continue

def gather_data(annotation_dir):
    """Gather data from a directory of CSV files.
    Args:
        annotation_dir (str): The directory containing the CSV files.
    
    Returns:
        pd.DataFrame: A DataFrame containing the data.
    """ 
    csvs = glob.glob(os.path.join(annotation_dir,"*.csv"))
    df = []
    for x in csvs:
        df.append(pd.read_csv(x))
    
    if len(df) == 0:
        return None
    df = pd.concat(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def get_api_key():
    """Get Label Studio API key from config file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               '.label_studio.config')
    if not os.path.exists(config_path):
        return None

    with open(config_path, 'r') as f:
        for line in f:
            if line.startswith('api_key'):
                return line.split('=')[1].strip()
    return None

def connect_to_label_studio(url, project_name, label_config=None):
    """Connect to the Label Studio server.
    Args:
        project_name (str): The name of the project to connect to.
        label_config (str, optional): The label configuration for the project. Defaults to None.
    Returns:
        str: The URL of the Label Studio server.
    """
    ls = Client(url=url, api_key=os.environ["LABEL_STUDIO_API_KEY"])
    ls.check_connection()

    # Look up existing name
    projects = ls.list_projects()
    project = [x for x in projects if x.get_params()["title"] == project_name]

    if len(project) == 0:
        # Create a project with the specified title and labeling configuration
        project = ls.create_project(
            title=project_name,
            label_config=label_config
        )
    else:
        project = project[0]

    return project

def create_project(ls, project_name):
    ls.create_project(title=project_name)

def create_sftp_client(user, host, key_filename):
    # Download annotations from Label Studio
    # SSH connection with a user prompt for password
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, key_filename=key_filename)
    sftp = ssh.open_sftp()

    return sftp

def delete_completed_tasks(label_studio_project):
    # Delete completed tasks
    tasks = label_studio_project.get_labeled_tasks()
    for task in tasks:
        label_studio_project.delete_task(task["id"])

def import_image_tasks(label_studio_project, image_names, local_image_dir, dataset_type, predictions=None, batch_size=10):
    """
    Import image tasks into Label Studio project in smaller batches.

    Args:
        label_studio_project (LabelStudioProject): The Label Studio project to import tasks into.
        image_names (list): List of image names to import as tasks.
        local_image_dir (str): The local directory where the images are stored.
        dataset_type (str): The type of dataset being imported.
        predictions (list, optional): List of predictions for each image. Defaults to None.
        batch_size (int, optional): Number of images to upload in each batch. Defaults to 10.

    Returns:
        None
    """
    # Process in batches
    for i in range(0, len(image_names), batch_size):
        batch_images = image_names[i:i + batch_size]
        tasks = []
        
        for j, image_name in enumerate(batch_images):
            print(f"Preparing {image_name} for Label Studio import")
            data_dict = {'image': os.path.join("/data/local-files/?d=input/", os.path.basename(image_name))}
            
            if predictions is not None:
                prediction = predictions[i + j]
                # Skip predictions if there are none
                if prediction.empty:
                    result_dict = []
                else:
                    result_dict = [label_studio_format(local_image_dir, prediction, dataset_type)]
                upload_dict = {"data": data_dict, "predictions": result_dict}
            else:
                upload_dict = {"data": data_dict}
            
            tasks.append(upload_dict)
        
        if tasks:
            try:
                print(f"Uploading batch {i//batch_size + 1} of {(len(image_names) + batch_size - 1)//batch_size}")
                label_studio_project.import_tasks(tasks)
            except Exception as e:
                print(f"Error uploading batch: {e}")
                continue

def download_completed_tasks(label_studio_project, csv_dir, dataset_type):
    """Download and convert completed tasks to pandas DataFrame"""
    labeled_tasks = label_studio_project.get_labeled_tasks()
    if not labeled_tasks:
        print("No new annotations")
        return None
    
    all_annotations = []
    for labeled_task in labeled_tasks:
        image_path = os.path.basename(labeled_task['data']['image'])
        label_json = labeled_task['annotations'][0]["result"]
        
        if len(label_json) == 0:
            # Handle empty annotations based on dataset type
            if dataset_type == 'TreePoints':
                result = pd.DataFrame({
                    "image_path": [image_path],
                    "x": [0],
                    "y": [0],
                    "label": [0],
                    "annotator": [labeled_task["annotations"][0]["created_username"]]
                })
            elif dataset_type == 'TreeBoxes':
                result = pd.DataFrame({
                    "image_path": [image_path],
                    "xmin": [0],
                    "ymin": [0],
                    "xmax": [0],
                    "ymax": [0],
                    "label": [0],
                    "annotator": [labeled_task["annotations"][0]["created_username"]]
                })
        else:
            result = convert_json_to_dataframe(label_json, dataset_type)
            result["image_path"] = image_path
            result["annotator"] = labeled_task["annotations"][0]["created_username"]
        
        all_annotations.append(result)

    annotations = pd.concat(all_annotations)
    print(f"There are {len(annotations)} new annotations")
    
    # Save csv with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(csv_dir, f"{dataset_type}_annotations_{timestamp}.csv")
    annotations.to_csv(csv_path, index=False)

    return annotations

def download_images(sftp_client, image_names, local_image_dir, folder_name):
    """
    Download images from a remote server using SFTP.

    Args:
        sftp_client (SFTPClient): An instance of the SFTPClient class representing the SFTP connection.
        image_names (list): A list of image file names to be downloaded.
        local_image_dir (str): The local directory where the images will be saved.
        folder_name (str): The name of the folder on the remote server where the images are located.

    Returns:
        None

    Raises:
        Any exceptions that may occur during the file transfer.

    """
    # SCP file transfer
    for image_name in image_names:
        remote_path = os.path.join(folder_name, "input", image_name)
        local_path = os.path.join(local_image_dir, image_name)
        try:
            sftp_client.get(remote_path, local_path)
        except FileNotFoundError:
            continue
        
        print(f"Downloaded {image_name} successfully")

def upload_images(sftp_client, images, folder_name):
    """
    Uploads a list of images to a remote server using SFTP.

    Args:
        sftp_client (SFTPClient): An instance of the SFTPClient class representing the SFTP connection.
        images (list): A list of image file paths to be uploaded.
        folder_name (str): The name of the folder on the remote server where the images will be uploaded.

    Returns:
        None

    Raises:
        Any exceptions that may occur during the file transfer.

    """
    # SCP file transfer
    for image in images:
        sftp_client.put(image, os.path.join(folder_name,"input",os.path.basename(image)))
        print(f"Uploaded {image} successfully")

def remove_annotated_images_remote_server(sftp_client, annotations, folder_name):
    """Remove images that have been annotated on the Label Studio server."""
    # Delete images using SSH
    for image in annotations.image_path.unique():
        remote_path = os.path.join(folder_name, "input", os.path.basename(image))
        # Archive annotations using SSH
        archive_annotation_path = os.path.join(folder_name, "archive", os.path.basename(image))
        # sftp check if dir exists
        try:
            sftp_client.listdir(os.path.join(folder_name, "archive"))
        except FileNotFoundError:
            raise FileNotFoundError("The archive directory {} does not exist.".format(os.path.join(folder_name, "archive")))
        
        sftp_client.rename(remote_path, archive_annotation_path)
        print(f"Archived {image} successfully")

