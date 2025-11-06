#!/bin/bash
#SBATCH --job-name=neon_unsupervised
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b.weinstein@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=/home/b.weinstein/logs/neon_unsupervised_%j.out
#SBATCH --error=/home/b.weinstein/logs/neon_unsupervised_%j.err

# Use uv for Python environment management
uv sync --extra dev

# Set working directory
cd /blue/ewhite/b.weinstein/src/MillionTrees/data_prep

# Configuration
DATA_DIR="/orange/ewhite/DeepForest/unsupervised"
CSV_GLOB="/blue/ewhite/veitchmichaelisj/deeplidar/output/*.csv"
FORMAT_OUTPUT_DIR="/orange/ewhite/DeepForest/unsupervised"
TOKEN_PATH="/blue/ewhite/b.weinstein/src/MillionTrees/neon_token.txt"
DOWNLOAD_DIR="/orange/ewhite/DeepForest/unsupervised"

# Create necessary directories
mkdir -p "$FORMAT_OUTPUT_DIR"
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$(dirname "$DATA_DIR")"

# Run the combined NEON unsupervised processing script
echo "Starting NEON unsupervised processing..."
echo "Data directory: $DATA_DIR"
echo "CSV glob pattern: $CSV_GLOB"
echo "Format output directory: $FORMAT_OUTPUT_DIR"
echo "Token path: $TOKEN_PATH"
echo "Download directory: $DOWNLOAD_DIR"

uv run process_neon_unsupervised.py \
    --csv_glob_pattern "$CSV_GLOB" \
    --format_output_dir "$FORMAT_OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --max_tiles_per_site 2 \
    --max_tiles 2 \
    --patch_size 400 \
    --allow_empty \
    --token_path "$TOKEN_PATH" \
    --data_product "DP3.30010.001" \
    --download_dir "$DOWNLOAD_DIR"

echo "NEON unsupervised processing completed!"

# Optional: Clean up download directory to save space
# echo "Cleaning up download directory..."
# rm -rf "$DOWNLOAD_DIR"

# Print summary statistics
echo "=== Summary Statistics ==="
if [ -f "$FORMAT_OUTPUT_DIR/TreeBoxes_neon_unsupervised.csv" ]; then
    echo "TreeBoxes annotations: $(wc -l < "$FORMAT_OUTPUT_DIR/TreeBoxes_neon_unsupervised.csv") lines"
fi

if [ -f "$FORMAT_OUTPUT_DIR/TreePoints_neon_unsupervised.csv" ]; then
    echo "TreePoints annotations: $(wc -l < "$FORMAT_OUTPUT_DIR/TreePoints_neon_unsupervised.csv") lines"
fi

if [ -f "$FORMAT_OUTPUT_DIR/TreePolygons_neon_unsupervised.csv" ]; then
    echo "TreePolygons annotations: $(wc -l < "$FORMAT_OUTPUT_DIR/TreePolygons_neon_unsupervised.csv") lines"
fi

if [ -f "$DATA_DIR/unsupervised/unsupervised_neon_tiled.parquet" ]; then
    echo "Tiled annotations created: $DATA_DIR/unsupervised/unsupervised_neon_tiled.parquet"
fi

if [ -d "$DATA_DIR/images" ]; then
    echo "Downloaded images: $(find "$DATA_DIR/images" -name "*.tif" | wc -l) TIF files"
fi

echo "Job completed at $(date)"