# HiPerGator Test Release Scripts

This directory contains scripts for running release validation tests on HiPerGator and generating dataset reports.

## Files

- **`submit_test_release.sh`** - SLURM submission script for HiPerGator
- **`generate_dataset_report.py`** - Standalone script to generate dataset size reports
- **`dataset_release_report.md`** - Generated report with dataset information

## Usage

### Running on HiPerGator

To submit the test release job to HiPerGator:

```bash
sbatch submit_test_release.sh
```

This will:
1. Run all tests in `tests/test_release.py`
2. Generate a comprehensive dataset report
3. Store logs and results in timestamped directories
4. Copy latest results to a standard location for easy access

### Generating Reports Locally

To generate a dataset report without running tests:

```bash
python generate_dataset_report.py
```

To include test results in the report:

```bash
python generate_dataset_report.py --test-exit-code 0  # for successful tests
python generate_dataset_report.py --test-exit-code 1  # for failed tests
```

To specify a custom output location:

```bash
python generate_dataset_report.py --output /path/to/report.md
```

## Output Locations

### HiPerGator Runs

- **Timestamped logs**: `/home/b.weinstein/logs/test_release_YYYYMMDD_HHMMSS/`
- **Latest results**: `/home/b.weinstein/logs/latest_test_release/`

### Generated Files

- `test_results.log` - Detailed pytest output
- `dataset_release_report.md` - Markdown report with dataset sizes
- `report_generation.log` - Log from report generation
- `status.txt` - Simple status summary

## Dataset Information

The report includes:

- **Summary table** with latest version and total sizes
- **Detailed version history** for each dataset
- **Download URLs** and compressed sizes
- **Test results** (when available)
- **Instructions** for running tests manually

### Current Dataset Sizes (v0.2)

- **TreePolygons**: ~70.24 GB
- **TreePoints**: ~1.36 GB  
- **TreeBoxes**: ~6.26 GB
- **Total**: ~77.86 GB

## Monitoring

Check job status on HiPerGator:

```bash
squeue -u $USER
```

View latest results:

```bash
cat /home/b.weinstein/logs/latest_test_release/status.txt
```

View the latest report:

```bash
cat /home/b.weinstein/logs/latest_test_release/dataset_release_report.md
```

## Scheduling

To run periodically, you can set up a cron job or use HiPerGator's job scheduling features to automatically validate releases.