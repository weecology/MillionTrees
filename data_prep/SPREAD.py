import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

DEBUG = False

def visualize_image(image, title=""):
    if DEBUG:
        plt.figure(figsize=(6, 6))
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.show()

def generate_coco_json(coco_data, output_path):
    """
    Save COCO annotation dictionary as JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    print(f"COCO JSON saved at {output_path}")

# ---------------------------
# Paths settings
# ---------------------------

rgb_image_path = "/Users/benweinstein/Downloads/aerial_birch_forest/rgb"
instance_image_path = "/Users/benweinstein/Downloads/aerial_birch_forest/instance_segmentation"
output_json_dir = "data/SPREAD"  # Output directory for COCO JSON files

# Path to color palette Excel file
color_palette_path = "/Users/benweinstein/Downloads/aerial_birch_forest/color_palette.xlsx"

# Ensure output directory exists
os.makedirs(output_json_dir, exist_ok=True)

# ---------------------------
# Read color palette and build mapping: color_id -> BGR color
# ---------------------------
df_palette = pd.read_excel(color_palette_path)
colorid_to_bgr = {}
for _, row in df_palette.iterrows():
    idx = int(row['Index'])
    # palette is in R,G,B order; convert to B,G,R for OpenCV
    R = int(row['R'])
    G = int(row['G'])
    B = int(row['B'])
    colorid_to_bgr[idx] = [B, G, R]

# List all image files (only .png)
rgb_images = sorted([f for f in os.listdir(rgb_image_path) if f.endswith(".png")])
instance_images = sorted([f for f in os.listdir(instance_image_path) if f.endswith(".png")])

# Process each image pair
# For COCO annotation, we need to assign unique image id and annotation id per image.
for img_idx in range(len(rgb_images)):
    np.random.seed(42)
    crown_colors = np.random.randint(0, 255, (1000, 3))
    crown_id = 0  # used to assign different colors to each detected crown (for visualization)

    # Full paths for images
    rgb_image_file = os.path.join(rgb_image_path, rgb_images[img_idx])
    instance_image_file = os.path.join(instance_image_path, instance_images[img_idx])

    # Define additional thresholds
    area_threshold = 625      # area filtering threshold for small instances
    min_cluster_size = 25     # minimum size for connected component in mask cleaning
    min_crown_size = 625      # minimum pixel count for a crown to be valid

    # ---------------------------
    # Read images
    # ---------------------------
    # Read RGB image (BGR by cv2, then convert to RGB)
    bgr_image = cv2.imread(rgb_image_file)
    if bgr_image is None:
        print(f"Cannot read RGB image: {rgb_image_file}")
        continue
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Read instance segmentation image (remains in BGR)
    instance_image = cv2.imread(instance_image_file)
    if instance_image is None:
        print(f"Cannot read instance segmentation image: {instance_image_file}")
        continue

    # ---------------------------
    # Read corresponding txt file to obtain valid color shortlist
    # ---------------------------
    txt_filename = os.path.splitext(os.path.basename(instance_image_file))[0] + ".txt"
    txt_path = os.path.join(instance_image_path, txt_filename)
    valid_color_list = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    # Expected format: "TreeID color_id"
                    if len(parts) >= 2:
                        try:
                            color_idx = int(parts[1])
                            if color_idx in colorid_to_bgr:
                                valid_color_list.append(tuple(colorid_to_bgr[color_idx]))
                        except Exception as e:
                            print(f"Error parsing line '{line}': {e}")
    else:
        print(f"TXT file not found for image {instance_image_file}. Using all colors.")
    # Remove duplicates by converting to set
    valid_color_set = set(valid_color_list)

    # ---------------------------
    # Filter out small area instances from instance image
    # ---------------------------
    def filter_small_instances(instance_img, area_thresh):
        filtered_instance_img = instance_img.copy()
        # Reshape to (num_pixels, 3) and get unique colors with their counts
        unique_colors, counts = np.unique(instance_img.reshape(-1, instance_img.shape[2]), axis=0, return_counts=True)
        for color, count in zip(unique_colors, counts):
            # Skip background: assume [0,0,0] is background
            if not np.array_equal(color, [0, 0, 0]):
                if count < area_thresh:
                    filtered_instance_img[(instance_img == color).all(axis=2)] = [0, 0, 0]
        return filtered_instance_img

    filtered_instance_image = filter_small_instances(instance_image, area_threshold)

    # Resize filtered instance image to match RGB image if needed
    if filtered_instance_image.shape[:2] != rgb_image.shape[:2]:
        filtered_instance_image = cv2.resize(filtered_instance_image, (rgb_image.shape[1], rgb_image.shape[0]))

    # Create an empty overlay for drawing polygons
    overlay = np.zeros_like(rgb_image, dtype=np.uint8)
    # List to store annotation features (for COCO)
    annotations = []

    # ---------------------------
    # Loop over unique colors in the filtered instance image
    # Only process colors that are in the valid_color_set (obtained from txt file)
    # ---------------------------
    unique_colors = np.unique(filtered_instance_image.reshape(-1, filtered_instance_image.shape[2]), axis=0)
    for color in unique_colors:
        color_tuple = tuple(int(v) for v in color)
        # Skip background and colors not in valid set
        if np.array_equal(color, [0, 0, 0]) or (valid_color_set and (color_tuple not in valid_color_set)):
            continue

        # Extract the mask for current color
        mask = cv2.inRange(filtered_instance_image, np.array(color, dtype=np.uint8), np.array(color, dtype=np.uint8))

        # Remove small connected components in the mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean_mask = np.zeros_like(mask)
        for label in range(1, num_labels):  # skip background label 0
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_cluster_size:
                clean_mask[labels == label] = 255
        mask = clean_mask

        visualize_image(mask, title="Clean Mask")

        # Copy mask for further processing
        test_mask = mask.copy()
        # Dilate the mask (kernel size 3x3, 15 iterations)
        kernel = np.ones((3, 3), np.uint8)
        test_mask = cv2.dilate(test_mask, kernel, iterations=15)
        visualize_image(test_mask, title="Dilated Mask")

        # Get connected components from dilated mask (using connectivity=4)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(test_mask, connectivity=4)
        visualize_image(labels.astype(np.uint8) * (255 // (num_labels+1)), title="Labels")

        # Create an image to accumulate different clusters with different colors
        test_mask_with_different_color = np.zeros_like(filtered_instance_image)
        # Process each connected component
        for label in range(1, num_labels):
            # Create a mask for current component
            single_component_mask = np.zeros_like(filtered_instance_image)
            single_component_mask[labels == label] = crown_colors[crown_id]
            # If the component is empty, skip
            if np.sum(single_component_mask) == 0:
                continue
            visualize_image(single_component_mask, title="Component Mask Before Erosion")
            # Erode the component (kernel 3x3, 13 iterations)
            kernel = np.ones((3, 3), np.uint8)
            single_component_mask = cv2.erode(single_component_mask, kernel, iterations=13)
            visualize_image(single_component_mask, title="Component Mask After Erosion")

            # Convert to grayscale for further processing
            gray_mask = cv2.cvtColor(single_component_mask, cv2.COLOR_RGB2GRAY)
            # Only keep the largest connected component
            comp_labels = cv2.connectedComponents(gray_mask, connectivity=4)[1]
            # Compute area for each label (skip background label 0)
            areas = [np.sum(comp_labels == i) for i in range(1, np.max(comp_labels)+1)]
            if len(areas) == 0:
                continue
            max_label = np.argmax(areas) + 1
            single_component_mask[comp_labels != max_label] = 0
            visualize_image(single_component_mask, title="Largest Component Only")
            # Check if the remaining area is large enough
            if np.sum(single_component_mask != 0) < min_crown_size*3:
                continue

            # Extract polygon from the current crown mask
            gray_single_crown = cv2.cvtColor(single_component_mask, cv2.COLOR_RGB2GRAY)
            contours, _ = cv2.findContours(gray_single_crown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = contours[0]
                # Compute bounding box for the contour
                x, y, w, h = cv2.boundingRect(contour)
                # Flatten contour coordinates for segmentation
                segmentation = [float(coord) for point in contour for coord in point[0]]
                # Compute area of the contour
                area = cv2.contourArea(contour)
                # Save annotation (COCO format) for this crown
                ann = {
                    "id": crown_id,
                    "image_id": img_idx,
                    "category_id": 1,  # category 1: tree
                    "segmentation": [segmentation],
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0
                }
                annotations.append(ann)

                # Draw the polygon overlay (filled) on the overlay image
                cv2.drawContours(overlay, [contour], -1, crown_colors[crown_id].tolist(), thickness=-1)
                # Also draw the bounding box (rectangle) in white (or use a contrasting color)
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 255), thickness=2)
            # Accumulate the component mask into the overall mask with different colors
            test_mask_with_different_color[single_component_mask != 0] = single_component_mask[single_component_mask != 0]

            crown_id += 1

        visualize_image(test_mask_with_different_color, title="Components with Different Colors")
        # Accumulate the colored mask into overlay (if not zero)
        overlay[test_mask_with_different_color != 0] = test_mask_with_different_color[test_mask_with_different_color != 0]

    # ---------------------------
    # Combine overlay with RGB image
    # ---------------------------
    alpha = 0.8
    combined_image = cv2.addWeighted(overlay, alpha, rgb_image, 1 - alpha, 0)

    # Additionally, visualize the combined image with bbox overlay separately if needed
    # For example, draw bbox on a copy of combined_image for clarity.
    combined_image_with_bbox = combined_image.copy()
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(combined_image_with_bbox, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    # Show final images (using matplotlib)
    fig, ax = plt.subplots(1, 3, figsize=(30, 8))
    ax[0].imshow(rgb_image)
    ax[0].set_title("Original RGB Image")
    ax[0].axis("off")
    ax[1].imshow(filtered_instance_image)
    ax[1].set_title("Instance Segmentation (After Filtering)")
    ax[1].axis("off")
    ax[2].imshow(combined_image_with_bbox)
    ax[2].set_title("RGB with Polygons & BBoxes Overlay")
    ax[2].axis("off")
    plt.show()

    # ---------------------------
    # Build COCO JSON structure for current image
    # ---------------------------
    image_info = {
        "id": img_idx,
        "file_name": os.path.basename(rgb_image_file),
        "width": rgb_image.shape[1],
        "height": rgb_image.shape[0]
    }
    coco_categories = [{
        "id": 1,
        "name": "tree",
        "supercategory": "tree"
    }]
    coco_data = {
        "images": [image_info],
        "annotations": annotations,
        "categories": coco_categories
    }

    # Save COCO JSON (one file per image)
    output_json_path = os.path.join(output_json_dir, f"{os.path.splitext(os.path.basename(rgb_image_file))[0]}.json")
    generate_coco_json(coco_data, output_json_path)

    # For demo
    if img_idx == 3:
        break