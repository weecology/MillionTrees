import glob
import random
import rasterio
from rasterio.windows import Window

def crop_random_image(glob_path, save_directory):
    # Search for image files using glob
    image_files = glob.glob(glob_path)

    if not image_files:
        print("No image files found.")
        return

    # Select a random image file
    random_image_file = random.choice(image_files)

    # Open the image file using rasterio
    with rasterio.open(random_image_file) as src:
        # Get the image dimensions
        height, width = src.shape

        # Generate random coordinates for the crop
        x = random.randint(0, width - 400)
        y = random.randint(0, height - 400)

        # Define the crop window
        window = Window(x, y, 400, 400)

        # Read the crop window from the image
        crop = src.read(window=window)

    # Save the cropped image to the specified directory
    save_path = f"{save_directory}/cropped_image.tif"
    with rasterio.open(save_path, 'w', **src.profile) as dst:
        dst.write(crop)

    print(f"Cropped image saved to: {save_path}")

# Example usage
glob_path = "path/to/images/*.tif"
save_directory = "path/to/save/directory"
crop_random_image(glob_path, save_directory)