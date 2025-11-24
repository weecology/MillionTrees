import os
import zipfile

def zip_directory(folder_path, zip_path):
    """Zip the contents of a directory."""
    # Remove the existing zip file if it exists
    if os.path.exists(zip_path):
        os.remove(zip_path)
    # Create a new zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

folder_paths = ["/orange/ewhite/web/public/MillionTrees/TreePolygons_v0.2", 
                "/orange/ewhite/web/public/MillionTrees/TreeBoxes_v0.2",
                "/orange/ewhite/web/public/MillionTrees/TreePoints_v0.2",
                "/orange/ewhite/web/public/MillionTrees/MiniTreePolygons_v0.2",
                "/orange/ewhite/web/public/MillionTrees/MiniTreeBoxes_v0.2",
                "/orange/ewhite/web/public/MillionTrees/MiniTreePoints_v0.2"]
zip_paths = [f"{path}.zip" for path in folder_paths]

for folder_path, zip_path in zip(folder_paths, zip_paths):
    zip_directory(folder_path, zip_path)

