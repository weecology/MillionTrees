from deepforest.preprocess import split_raster
import glob
import rasterio as rio

# Load images from folder
#files = glob.glob("/Users/benweinstein/Downloads/OneDrive_1_2-7-2024/*.tif")
files = ["/Users/benweinstein/Downloads/Plot13Ortho.tif"]
for f in files:
    # Open raster, select first three bands and scale to 0 to 255
    with rio.open(f) as src:
        bands = src.read()
        bands = bands / bands.max(axis=(1, 2)).reshape(-1, 1, 1)
        bands = (bands * 255).astype("uint8")
        bands = bands[:3, :, :]

    split_raster(
        numpy_image=bands,
        base_dir="/Users/benweinstein/Downloads/OneDrive_1_2-7-2024/crops/",
        patch_size=800,
        patch_overlap=0,
        image_name=f.split("/")[-1].split(".")[0]
        )

