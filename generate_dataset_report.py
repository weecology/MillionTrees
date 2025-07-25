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

# Add the src directory to the path to import milliontrees modules
repo_root = Path(__file__).parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

try:
    from milliontrees.datasets.TreePolygons import TreePolygonsDataset
    from milliontrees.datasets.TreePoints import TreePointsDataset  
    from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
except ImportError as e:
    print(f"Error importing milliontrees modules: {e}")
    print("Make sure you're running this from the MillionTrees repository root")
    sys.exit(1)


def format_bytes(bytes_size):
    """Convert bytes to human readable format."""
    if bytes_size == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    size = bytes_size
    i = 0
    
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.2f} {size_names[i]}"


def generate_dataset_report(output_path=None, include_test_results=False, test_exit_code=None):
    """Generate a comprehensive dataset report."""
    
    if output_path is None:
        output_path = repo_root / "dataset_release_report.md"
    
    # Get dataset version information
    datasets_info = [
        ('TreePolygons', TreePolygonsDataset),
        ('TreePoints', TreePointsDataset),
        ('TreeBoxes', TreeBoxesDataset)
    ]

    report_content = []
    report_content.append('# MillionTrees Dataset Release Report')
    report_content.append(f'**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report_content.append('')
    
    # Summary table
    report_content.append('## Dataset Summary')
    report_content.append('')
    report_content.append('| Dataset | Latest Version | Compressed Size | Download URL |')
    report_content.append('|---------|----------------|-----------------|--------------|')
    
    total_size = 0
    
    for dataset_name, dataset_class in datasets_info:
        versions_dict = getattr(dataset_class, '_versions_dict', {})
        
        if versions_dict:
            # Get latest version (highest version number)
            latest_version = max(versions_dict.keys(), key=lambda v: tuple(map(int, v.split('.'))))
            latest_info = versions_dict[latest_version]
            size_bytes = latest_info.get('compressed_size', 0)
            total_size += size_bytes
            url = latest_info.get('download_url', 'N/A')
            
            # Truncate URL for display
            display_url = url
            if len(url) > 50:
                display_url = url[:47] + "..."
            
            report_content.append(f'| {dataset_name} | {latest_version} | {format_bytes(size_bytes)} | {display_url} |')
        else:
            report_content.append(f'| {dataset_name} | N/A | N/A | N/A |')
    
    report_content.append(f'| **Total** | - | **{format_bytes(total_size)}** | - |')
    report_content.append('')

    # Detailed version information
    report_content.append('## Detailed Dataset Information')
    report_content.append('')

    for dataset_name, dataset_class in datasets_info:
        report_content.append(f'### {dataset_name}')
        report_content.append('')
        
        # Add dataset description if available
        doc_string = getattr(dataset_class, '__doc__', '')
        if doc_string:
            # Extract the first paragraph of the docstring
            first_paragraph = doc_string.strip().split('\n\n')[0]
            # Clean up the docstring formatting
            first_paragraph = ' '.join(line.strip() for line in first_paragraph.split('\n'))
            report_content.append(f'**Description:** {first_paragraph}')
            report_content.append('')
        
        versions_dict = getattr(dataset_class, '_versions_dict', {})
        
        if versions_dict:
            report_content.append('| Version | Download URL | Compressed Size | Size (MB) | Size (GB) |')
            report_content.append('|---------|-------------|-----------------|-----------|-----------|')
            
            # Sort versions by version number
            sorted_versions = sorted(versions_dict.keys(), key=lambda v: tuple(map(int, v.split('.'))))
            
            for version in sorted_versions:
                info = versions_dict[version]
                url = info.get('download_url', 'N/A')
                size_bytes = info.get('compressed_size', 0)
                size_mb = round(size_bytes / (1024 * 1024), 2)
                size_gb = round(size_bytes / (1024 * 1024 * 1024), 2)
                
                report_content.append(f'| {version} | {url} | {size_bytes:,} bytes | {size_mb} | {size_gb} |')
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
    
    args = parser.parse_args()
    
    include_test_results = args.test_exit_code is not None
    generate_dataset_report(
        output_path=args.output,
        include_test_results=include_test_results,
        test_exit_code=args.test_exit_code
    )