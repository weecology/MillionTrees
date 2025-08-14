import numpy as np
from typing import Dict, Any, List, Optional


def format_eval_results(results: Dict[str, Any], dataset) -> str:
    """
    Format evaluation results into well-formatted tables.
    
    Args:
        results: Dictionary containing evaluation results
        dataset: Dataset object with source mapping information
        
    Returns:
        Formatted string with tables
    """
    formatted_output = []
    
    # Get source name mapping if available
    source_names = {}
    if hasattr(dataset, '_source_id_to_code'):
        source_names = dataset._source_id_to_code
    
    for metric_name, metric_results in results.items():
        if not isinstance(metric_results, dict):
            continue
            
        formatted_output.append(f"\n{'='*60}")
        formatted_output.append(f"{metric_name.upper()} RESULTS")
        formatted_output.append(f"{'='*60}")
        
        # Extract source-specific results
        source_data = []
        avg_score = None
        worst_group_score = None
        
        for key, value in metric_results.items():
            if key.endswith('_avg'):
                avg_score = value
            elif key.endswith('_wg'):
                worst_group_score = value
            elif '_group:' in key and not key.startswith('count_'):
                # Extract group ID and score
                group_id = int(key.split(':')[1])
                source_name = source_names.get(group_id, f"Source {group_id}")
                count_key = f'count_group:{group_id}'
                count = metric_results.get(count_key, 0)
                
                if not np.isnan(value) and count > 0:
                    source_data.append({
                        'Source': source_name,
                        'Score': round(value, 3),
                        'Count': int(count)
                    })
        
        # Create source table
        if source_data:
            # Sort by score descending
            source_data.sort(key=lambda x: x['Score'], reverse=True)
            
            formatted_output.append("\nSource-wise Results:")
            formatted_output.append("-" * 50)
            formatted_output.append(f"{'Source':<25} {'Score':<10} {'Count':<10}")
            formatted_output.append("-" * 50)
            
            for row in source_data:
                formatted_output.append(f"{row['Source']:<25} {row['Score']:<10.3f} {row['Count']:<10}")
        
        # Add summary statistics
        formatted_output.append("\nSummary Statistics:")
        formatted_output.append("-" * 40)
        if avg_score is not None and not np.isnan(avg_score):
            formatted_output.append(f"Average {metric_name}: {avg_score:.3f}")
        if worst_group_score is not None and not np.isnan(worst_group_score):
            formatted_output.append(f"Worst-group {metric_name}: {worst_group_score:.3f}")
        
        if source_data:
            scores = [row['Score'] for row in source_data]
            formatted_output.append(f"Min {metric_name}: {min(scores):.3f}")
            formatted_output.append(f"Max {metric_name}: {max(scores):.3f}")
            formatted_output.append(f"Std {metric_name}: {np.std(scores):.3f}")
    
    return "\n".join(formatted_output)


def format_eval_results_simple(results: Dict[str, Any], dataset) -> str:
    """
    A simpler version that just formats the results without creating tables.
    
    Args:
        results: Dictionary containing evaluation results
        dataset: Dataset object with source mapping information
        
    Returns:
        Formatted string
    """
    formatted_output = []
    
    # Get source name mapping if available
    source_names = {}
    if hasattr(dataset, '_source_id_to_code'):
        source_names = dataset._source_id_to_code
    
    for metric_name, metric_results in results.items():
        if not isinstance(metric_results, dict):
            continue
            
        formatted_output.append(f"\n{metric_name.upper()}:")
        formatted_output.append("-" * 40)
        
        # Extract source-specific results
        for key, value in metric_results.items():
            if '_group:' in key and not key.startswith('count_'):
                group_id = int(key.split(':')[1])
                source_name = source_names.get(group_id, f"Source {group_id}")
                count_key = f'count_group:{group_id}'
                count = metric_results.get(count_key, 0)
                
                if not np.isnan(value) and count > 0:
                    formatted_output.append(f"{source_name}: {value:.3f} (n={int(count)})")
        
        # Add summary
        avg_key = f'{metric_name}_avg'
        wg_key = f'{metric_name}_wg'
        
        if avg_key in metric_results and not np.isnan(metric_results[avg_key]):
            formatted_output.append(f"Average: {metric_results[avg_key]:.3f}")
        if wg_key in metric_results and not np.isnan(metric_results[wg_key]):
            formatted_output.append(f"Worst-group: {metric_results[wg_key]:.3f}")
    
    return "\n".join(formatted_output) 