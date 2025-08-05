#!/bin/bash
#SBATCH --job-name=MillionTrees_test_release   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=ben.weinstein@weecology.org # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/MillionTrees_test_release_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/MillionTrees_test_release_%j.err
#SBATCH --gpus=1

source /blue/ewhite/b.weinstein/miniconda3/etc/profile.d/conda.sh
conda activate MillionTrees

# Add to path
export PYTHONPATH=$PYTHONPATH:/home/b.weinstein/MillionTrees

# Create a timestamped log directory
LOG_DIR="/home/b.weinstein/logs/test_release_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Change to the MillionTrees directory
cd /home/b.weinstein/MillionTrees

echo "Starting MillionTrees test_release validation at $(date)"
echo "Log directory: $LOG_DIR"

# Run the test_release.py tests with verbose output and capture results
python -m pytest tests/test_release.py -v --tb=long --capture=no 2>&1 | tee $LOG_DIR/test_results.log

# Capture the exit code from pytest
PYTEST_EXIT_CODE=${PIPESTATUS[0]}

# Generate dataset size report
echo "Generating dataset size report..."
python generate_dataset_report.py --output $LOG_DIR/dataset_release_report.md --test-exit-code $PYTEST_EXIT_CODE 2>&1 | tee $LOG_DIR/report_generation.log

# Check if report generation was successful
if [ -f "$LOG_DIR/dataset_release_report.md" ]; then
    echo "Dataset report generated successfully"
else
    echo "Warning: Dataset report generation may have failed"
fi

# Also generate a simple status file for easy checking
echo "Test run completed at $(date)" > $LOG_DIR/status.txt
echo "Pytest exit code: $PYTEST_EXIT_CODE" >> $LOG_DIR/status.txt
echo "Report generation status: $([ -f "$LOG_DIR/dataset_release_report.md" ] && echo "SUCCESS" || echo "FAILED")" >> $LOG_DIR/status.txt

# Copy the latest report to a standard location for easy access
LATEST_REPORT_DIR="/home/b.weinstein/logs/latest_test_release"
mkdir -p $LATEST_REPORT_DIR

# Copy files if they exist
if [ -f "$LOG_DIR/dataset_release_report.md" ]; then
    cp $LOG_DIR/dataset_release_report.md $LATEST_REPORT_DIR/
fi
if [ -f "$LOG_DIR/test_results.log" ]; then
    cp $LOG_DIR/test_results.log $LATEST_REPORT_DIR/
fi
if [ -f "$LOG_DIR/status.txt" ]; then
    cp $LOG_DIR/status.txt $LATEST_REPORT_DIR/
fi

echo "Test release validation completed at $(date)"
echo "Results available in: $LOG_DIR"
echo "Latest results copied to: $LATEST_REPORT_DIR"

# Exit with the same code as pytest for proper SLURM status reporting
exit $PYTEST_EXIT_CODE
