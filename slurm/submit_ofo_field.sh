#!/bin/bash
#SBATCH --job-name=ofo_field
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=b.weinstein@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/b.weinstein/logs/ofo_field_%j.out
#SBATCH --error=/home/b.weinstein/logs/ofo_field_%j.err

uv sync --group dev

cd /blue/ewhite/b.weinstein/src/MillionTrees/data_prep

# Path to the concatenated OFO field-trees gpkg shared by David Young (330 plot-drone pairs).
# Update this to wherever the gpkg is staged on Orange.
FIELD_TREES="/orange/ewhite/DeepForest/OpenForestObservatory/field/field_trees.gpkg"

OUTPUT_DIR="/orange/ewhite/DeepForest/OpenForestObservatory/field"
OFO_ROOT="/orange/ewhite/DeepForest/OpenForestObservatory/missions_03"

mkdir -p "$OUTPUT_DIR" "$OFO_ROOT"

echo "Starting OFO field-trees processing..."
echo "Field trees: $FIELD_TREES"
echo "Output directory: $OUTPUT_DIR"
echo "OFO mission cache: $OFO_ROOT"

uv run process_ofo_field.py \
    --field_trees "$FIELD_TREES" \
    --output_dir "$OUTPUT_DIR" \
    --ofo_root "$OFO_ROOT" \
    --patch_size 800 \
    --sample_plots

echo "OFO field-trees processing completed!"

if [ -f "$OUTPUT_DIR/TreePoints_OFO_field.csv" ]; then
    echo "Annotations rows: $(wc -l < "$OUTPUT_DIR/TreePoints_OFO_field.csv")"
fi
if [ -d "$OUTPUT_DIR/images" ]; then
    echo "Tiled images: $(find "$OUTPUT_DIR/images" -name '*.png' | wc -l)"
fi

echo "Job completed at $(date)"
