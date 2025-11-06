#!/bin/bash
#SBATCH --job-name=ofo_unsupervised
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b.weinstein@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH --time=12:00:00
#SBATCH --output=/home/b.weinstein/logs/ofo_unsupervised_%j.out
#SBATCH --error=/home/b.weinstein/logs/ofo_unsupervised_%j.err

uv sync --extra dev

# Set working directory
cd /blue/ewhite/b.weinstein/src/MillionTrees/data_prep

# Configuration
DATA_DIR="/orange/ewhite/DeepForest/OpenForestObservatory"
OUTPUT_DIR="/orange/ewhite/DeepForest/OpenForestObservatory/unsupervised"
OFO_ROOT="/orange/ewhite/DeepForest/OpenForestObservatory"

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OFO_ROOT"
mkdir -p "$(dirname "$DATA_DIR")"

# Run the combined OFO unsupervised processing script
echo "Starting OFO unsupervised processing..."
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "OFO download root: $OFO_ROOT"

uv run process_ofo_unsupervised.py \
    --data_dir "$DATA_DIR" \
    --ofo_root "$OFO_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --patch_size 800 \
    --num_missions 2 \

echo "OFO unsupervised processing completed!"

# Optional: Clean up download directory to save space
# echo "Cleaning up download directory..."
# rm -rf "$OFO_ROOT"

# Print summary statistics
echo "=== Summary Statistics ==="
if [ -f "$OUTPUT_DIR/points/TreePoints_OFO.csv" ]; then
    echo "TreePoints annotations: $(wc -l < "$OUTPUT_DIR/points/TreePoints_OFO.csv") lines"
fi

if [ -f "$OUTPUT_DIR/boxes/TreeBoxes_OFO.csv" ]; then
    echo "TreeBoxes annotations: $(wc -l < "$OUTPUT_DIR/boxes/TreeBoxes_OFO.csv") lines"
fi

if [ -f "$DATA_DIR/unsupervised/TreePoints_OFO_unsupervised.parquet" ]; then
    echo "Tiled annotations created: $DATA_DIR/unsupervised/TreePoints_OFO_unsupervised.parquet"
fi

if [ -d "$DATA_DIR/images" ]; then
    echo "Downloaded images: $(find "$DATA_DIR/images" -name "*.png" | wc -l) PNG files"
fi

echo "Job completed at $(date)"