#!/usr/bin/env python3
"""
Generate a dataset release report for MillionTrees datasets.

This script extracts dataset size information from the dataset classes
and generates a markdown report with current dataset versions and sizes.
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add the src directory to the path to import milliontrees modules
repo_root = Path(__file__).parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

try:
    from milliontrees.datasets.TreePolygons import TreePolygonsDataset
    from milliontrees.datasets.TreePoints import TreePointsDataset
    from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
    DATASET_IMPORTS_AVAILABLE = True
except ImportError:
    # Fallback minimal metadata for report generation without full dependencies
    DATASET_IMPORTS_AVAILABLE = False
    TreePolygonsDataset = {
        "description": "The TreePolygons dataset is a collection of tree annotations annotated as multi-point polygon locations.",
        "versions": {
            "0.8": {
                "download_url": "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_v0.8.zip",
                "compressed_size": 0,
            }
        },
    }
    TreePointsDataset = {
        "description": "The TreePoints dataset is a collection of tree annotations annotated as x,y locations.",
        "versions": {
            "0.8": {
                "download_url": "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_v0.8.zip",
                "compressed_size": 0,
            }
        },
    }
    TreeBoxesDataset = {
        "description": "A dataset of tree annotations with bounding box coordinates from multiple global sources.",
        "versions": {
            "0.8": {
                "download_url": "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreeBoxes_v0.8.zip",
                "compressed_size": 0,
            }
        },
    }


def get_versions_dict(dataset_obj) -> dict:
    """Extract versions dict from either a dataset class or fallback dict."""
    if hasattr(dataset_obj, "_versions_dict"):
        return getattr(dataset_obj, "_versions_dict")
    if isinstance(dataset_obj, dict):
        return dataset_obj.get("versions", {})
    return {}


def get_description(dataset_obj) -> str:
    """Extract description from dataset class docstring or fallback dict."""
    if hasattr(dataset_obj, "__doc__") and getattr(dataset_obj, "__doc__"):
        doc_string = getattr(dataset_obj, "__doc__", "")
        first_paragraph = doc_string.strip().split('\n\n')[0]
        return ' '.join(line.strip() for line in first_paragraph.split('\n'))
    if isinstance(dataset_obj, dict):
        return dataset_obj.get("description", "")
    return ""

def format_gb(bytes_size: int) -> str:
    """Convert bytes to GB with two decimals."""
    if not isinstance(bytes_size, (int, float)) or bytes_size <= 0:
        return "0.00 GB"
    return f"{bytes_size / (1024 ** 3):.2f} GB"


def derive_local_path_from_url(url: str) -> Optional[Path]:
    """Map UFRC HTTPS URL to on-disk path if available."""
    if not url or "data.rc.ufl.edu/pub/" not in url:
        return None
    # Map https://data.rc.ufl.edu/pub/... -> /orange/ewhite/web/public/...
    tail = url.split("data.rc.ufl.edu/pub/", 1)[1]
    # Drop leading 'ewhite/' since on-disk path starts at /orange/ewhite/web/public/MillionTrees/...
    if tail.startswith("ewhite/"):
        tail = tail[len("ewhite/"):]
    return Path("/orange/ewhite/web/public") / tail


def get_on_disk_size_bytes(url: str, fallback_bytes: int) -> int:
    """Return on-disk size in bytes if file exists, otherwise fallback."""
    local_path = derive_local_path_from_url(url)
    if local_path and local_path.exists():
        return local_path.stat().st_size
    return fallback_bytes or 0


def derive_supervised_url(url: str) -> Optional[str]:
    """Given a full-dataset URL like .../TreeBoxes_v0.11.zip, return the supervised variant URL."""
    if not url:
        return None
    import re
    # Insert '_supervised' before the version string: TreeBoxes_v0.11.zip -> TreeBoxes_supervised_v0.11.zip
    return re.sub(r'(Tree(?:Boxes|Points|Polygons))(_v[\d.]+\.zip)', r'\1_supervised\2', url)


def get_unsupervised_size_bytes(full_url: str) -> int:
    """Return size in bytes attributable to unsupervised images (full zip minus supervised zip)."""
    full_bytes = get_on_disk_size_bytes(full_url, 0)
    sup_url = derive_supervised_url(full_url)
    sup_bytes = get_on_disk_size_bytes(sup_url, 0) if sup_url else 0
    return max(0, full_bytes - sup_bytes)


def read_random_csv_counts_from_zip(url: str) -> Optional[dict]:
    """Open the zip on disk (mapped from URL) and count images/annotations/sources from random.csv.

    Returns counts split into supervised (sources without 'unsupervised' or 'weak supervised')
    and unsupervised (sources whose names contain 'unsupervised' or 'weak supervised').
    """
    import io
    import csv
    import zipfile
    import re

    _unsupervised_re = re.compile(r"unsupervised|weak supervised", re.IGNORECASE)

    local_path = derive_local_path_from_url(url)
    if not local_path or not local_path.exists():
        return None
    try:
        with zipfile.ZipFile(local_path, "r") as zf:
            names = zf.namelist()
            candidates = [n for n in names if n.lower().endswith("random.csv")]
            if not candidates:
                return None
            # Prefer the shallowest path
            name = sorted(candidates, key=lambda s: s.count("/"))[0]
            with zf.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                sup_rows = 0
                sup_filenames: set = set()
                sup_sources: set = set()
                unsup_rows = 0
                unsup_filenames: set = set()
                unsup_sources: set = set()
                for row in reader:
                    fn = row.get("filename")
                    src = row.get("source", "") or ""
                    if _unsupervised_re.search(src):
                        unsup_rows += 1
                        if fn:
                            unsup_filenames.add(fn)
                        if src:
                            unsup_sources.add(src)
                    else:
                        sup_rows += 1
                        if fn:
                            sup_filenames.add(fn)
                        if src:
                            sup_sources.add(src)
                return {
                    "images": len(sup_filenames),
                    "annotations": sup_rows,
                    "sources": len(sup_sources),
                    "unsup_images": len(unsup_filenames),
                    "unsup_annotations": unsup_rows,
                    "unsup_sources": len(unsup_sources),
                }
    except zipfile.BadZipFile:
        # Some URLs may point to non-zip resources; treat as missing stats.
        return None


def read_split_image_counts_from_zip(url: str, split_basename: str) -> Optional[dict]:
    """Count unique images in train/test from <split_basename>.csv inside the zip."""
    import io
    import csv
    import zipfile
    local_path = derive_local_path_from_url(url)
    if not local_path or not local_path.exists():
        return None
    try:
        with zipfile.ZipFile(local_path, "r") as zf:
            names = zf.namelist()
            candidates = [n for n in names if n.lower().endswith(f"{split_basename}.csv")]
            if not candidates:
                return None
            name = sorted(candidates, key=lambda s: s.count("/"))[0]
            with zf.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                train_files = set()
                test_files = set()
                for row in reader:
                    fn = row.get("filename")
                    split = (row.get("split") or "").strip().lower()
                    if not fn:
                        continue
                    if split == "train":
                        train_files.add(fn)
                    elif split == "test":
                        test_files.add(fn)
                return {"train_images": len(train_files), "test_images": len(test_files)}
    except zipfile.BadZipFile:
        # Non-zip resources should not cause report generation to fail.
        return None


def generate_dataset_report(output_path=None, include_test_results=False, test_exit_code=None, output_to_docs=True):
    """Generate a comprehensive dataset report."""
    
    if output_path is None:
        if output_to_docs:
            output_path = repo_root / "docs" / "dataset_release_report.md"
        else:
            output_path = repo_root / "dataset_release_report.md"
    
    # Get dataset version information
    datasets_info = [
        ('TreePolygons', TreePolygonsDataset),
        ('TreePoints', TreePointsDataset),
        ('TreeBoxes', TreeBoxesDataset),
    ]

    report_content = []
    report_content.append('# MillionTrees Dataset Release Report')
    report_content.append(f'**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report_content.append('')
    
    # Summary table
    report_content.append('## Dataset Summary (Supervised Sources Only)')
    report_content.append('')
    report_content.append('| Dataset | Latest Version | Size (GB) | Images | Annotations | Sources | Download URL |')
    report_content.append('|---------|----------------|-----------|--------|-------------|---------|--------------|')

    total_size = 0
    all_counts = {}  # cache for reuse in detailed section
    all_urls = {}    # full dataset URL per dataset name

    def _version_sort_key(v):
        try:
            return tuple(map(int, v.split('.')))
        except ValueError:
            return (0, 0)

    for dataset_name, dataset_class in datasets_info:
        versions_dict = get_versions_dict(dataset_class)

        if versions_dict:
            latest_version = max(versions_dict.keys(), key=_version_sort_key)
            latest_info = versions_dict[latest_version]
            url = latest_info.get('download_url', 'N/A')
            size_bytes_meta = latest_info.get('compressed_size', 0)
            try:
                size_bytes_meta = int(size_bytes_meta)
            except (ValueError, TypeError):
                size_bytes_meta = 0
            size_bytes_int = get_on_disk_size_bytes(url, size_bytes_meta)
            total_size += size_bytes_int
            counts = read_random_csv_counts_from_zip(url) or {
                "images": 0, "annotations": 0, "sources": 0,
                "unsup_images": 0, "unsup_annotations": 0, "unsup_sources": 0,
            }
            all_counts[dataset_name] = counts
            all_urls[dataset_name] = url

            # Handle missing URLs
            if not url or url.strip() == '':
                url = 'N/A'

            # Truncate URL for display
            display_url = url
            if url != 'N/A' and len(url) > 50:
                display_url = url[:47] + "..."
            report_content.append(
                f'| {dataset_name} | {latest_version} | {format_gb(size_bytes_int)} | '
                f'{counts["images"]} | {counts["annotations"]} | {counts["sources"]} | {display_url} |'
            )
        else:
            report_content.append(f'| {dataset_name} | N/A | N/A | N/A | N/A | N/A | N/A |')
            all_counts[dataset_name] = {}

    report_content.append(f'| **Total** | - | **{format_gb(total_size)}** | - | - | - | - |')
    report_content.append('')

    # Unsupervised sources summary (train-only, separate table)
    unsup_rows = []
    for dataset_name, counts in all_counts.items():
        if counts.get("unsup_images", 0) > 0 or counts.get("unsup_annotations", 0) > 0:
            unsup_rows.append((
                dataset_name,
                counts.get("unsup_images", 0),
                counts.get("unsup_annotations", 0),
                counts.get("unsup_sources", 0),
            ))
    if unsup_rows:
        report_content.append('## Unsupervised Sources (Train Split Only)')
        report_content.append('')
        report_content.append('Sources with "unsupervised" or "weak supervised" in their name are excluded from test '
                               'splits and counted separately here. Size is estimated as full zip minus supervised zip.')
        report_content.append('')
        report_content.append('| Dataset | Size (GB) | Images | Annotations | Sources |')
        report_content.append('|---------|-----------|--------|-------------|---------|')
        for row in unsup_rows:
            unsup_size = get_unsupervised_size_bytes(all_urls.get(row[0], ''))
            report_content.append(f'| {row[0]} | {format_gb(unsup_size)} | {row[1]} | {row[2]} | {row[3]} |')
        report_content.append('')

    # Detailed version information
    report_content.append('## Detailed Dataset Information')
    report_content.append('')

    for dataset_name, dataset_class in datasets_info:
        report_content.append(f'### {dataset_name}')
        report_content.append('')

        # Description intentionally omitted in the detailed section

        versions_dict = get_versions_dict(dataset_class)

        if versions_dict:
            report_content.append('| Version | Download URL | Size (GB) |')
            report_content.append('|---------|-------------|-----------|')

            sorted_versions = sorted(versions_dict.keys(), key=_version_sort_key)

            for version in sorted_versions:
                info = versions_dict[version]
                url = info.get('download_url', 'N/A')
                size_bytes_meta = info.get('compressed_size', 0)
                try:
                    size_bytes_meta = int(size_bytes_meta)
                except (ValueError, TypeError):
                    size_bytes_meta = 0
                size_bytes_int = get_on_disk_size_bytes(url, size_bytes_meta)
                # Handle missing URLs
                if not url or url.strip() == '':
                    url = 'N/A'
                report_content.append(f'| {version} | {url} | {format_gb(size_bytes_int)} |')

            # Add latest version dataset stats (supervised sources only)
            latest_version = sorted_versions[-1]
            latest_url = versions_dict[latest_version].get('download_url', '')
            counts = all_counts.get(dataset_name)
            if counts:
                report_content.append('')
                report_content.append('**Latest Version Dataset Stats (random split, supervised sources only):**')
                report_content.append('')
                report_content.append(f"- Images: {counts['images']}")
                report_content.append(f"- Annotations: {counts['annotations']}")
                report_content.append(f"- Sources: {counts['sources']}")

            # Add train/test split counts for random and zeroshot (images)
            split_rows = []
            for split_name in ("random", "zeroshot"):
                split_counts = read_split_image_counts_from_zip(latest_url, split_name)
                if split_counts:
                    split_rows.append((split_name, split_counts["train_images"], split_counts["test_images"]))
            if split_rows:
                report_content.append('')
                report_content.append('**Split counts (images):**')
                report_content.append('')
                report_content.append('| Split | Train | Test |')
                report_content.append('|-------|-------|------|')
                for name, tr, te in split_rows:
                    report_content.append(f'| {name} | {tr} | {te} |')
        else:
            report_content.append('No version information available.')
        
        report_content.append('')

    # Add test results if provided
    if include_test_results:
        report_content.append('## Test Results Summary')
        report_content.append('')
        
        if test_exit_code == 0:
            report_content.append('✅ **All tests passed successfully**')
        elif test_exit_code is not None:
            report_content.append(f'❌ **Tests failed with exit code:** {test_exit_code}')
        else:
            report_content.append('⚠️ **Test results not available**')
        
        report_content.append('')

    # Add information about the release tests
    report_content.append('## Release Test Information')
    report_content.append('')
    report_content.append('The release tests validate that:')
    report_content.append('- Datasets can be downloaded successfully')
    report_content.append('- Dataset structure and format are correct')
    report_content.append('- Image and annotation data have expected shapes and types')
    report_content.append('- Data loaders work properly for training and evaluation')
    report_content.append('')
    report_content.append('Tests are located in: `tests/test_release.py`')
    report_content.append('')
    report_content.append('To run the tests manually:')
    report_content.append('```bash')
    report_content.append('python -m pytest tests/test_release.py -v')
    report_content.append('```')
    report_content.append('')

    # Write the report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_content))

    print(f'Dataset report generated: {output_path}')
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate MillionTrees dataset release report')
    parser.add_argument('--output', '-o', type=str, help='Output file path for the report')
    parser.add_argument('--test-exit-code', type=int, help='Exit code from test run to include in report')
    parser.add_argument('--docs', action='store_true', default=True, help='Generate report in docs directory (default)')
    parser.add_argument('--root', action='store_true', help='Generate report in root directory')
    
    args = parser.parse_args()
    
    # Determine output location
    output_to_docs = not args.root  # Default to docs unless --root is specified
    
    include_test_results = args.test_exit_code is not None
    generate_dataset_report(
        output_path=args.output,
        include_test_results=include_test_results,
        test_exit_code=args.test_exit_code,
        output_to_docs=output_to_docs
    )