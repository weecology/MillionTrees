import torch
import numpy as np
from torch.utils.data import Subset
from pandas.api.types import CategoricalDtype
from typing import Dict, Any, List, Optional


def minimum(numbers, empty_val=0.):
    if isinstance(numbers, torch.Tensor):
        if numbers.numel() == 0:
            return torch.tensor(empty_val, device=numbers.device)
        else:
            return numbers[~torch.isnan(numbers)].min()
    elif isinstance(numbers, np.ndarray):
        if numbers.size == 0:
            return np.array(empty_val)
        else:
            return np.nanmin(numbers)
    elif len(numbers) == 0:
        return empty_val
    else:
        return min(numbers)


def maximum(numbers, empty_val=0.):
    if isinstance(numbers, torch.Tensor):
        if numbers.numel() == 0:
            return torch.tensor(empty_val, device=numbers.device)
        else:
            return numbers[~torch.isnan(numbers)].max()
    elif isinstance(numbers, np.ndarray):
        if numbers.size == 0:
            return np.array(empty_val)
        else:
            return np.nanmax(numbers)
    elif len(numbers) == 0:
        return empty_val
    else:
        return max(numbers)


def split_into_groups(g):
    """Splits the input tensor into unique groups and their corresponding indices.

    Args:
        g (Tensor): A vector containing group labels.

    Returns:
        tuple:
            - groups (Tensor): A tensor containing the unique group labels present in `g`.
            - group_indices (list of Tensors): A list where each tensor contains the indices
              of elements in `g` that correspond to the respective group in `groups`.
            - unique_counts (Tensor): A tensor representing the count of each unique group
              in `groups`, with the same length as `groups`.
    """

    unique_groups, unique_counts = torch.unique(g,
                                                sorted=False,
                                                return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts


def get_counts(g, n_groups):
    """This differs from split_into_groups in how it handles missing groups. get_counts always
    returns a count array of length n_groups, whereas split_into_groups returns a unique_counts
    array whose length is the number of unique groups present in g.

    Args:
        - g (ndarray): Vector of groups
    Returns:
        - counts (ndarray): An array of length n_groups, denoting the count of each group.
    """
    unique_groups, unique_counts = torch.unique(g,
                                                sorted=False,
                                                return_counts=True)
    counts = torch.zeros(n_groups, device=g.device)
    counts[unique_groups] = unique_counts.float()

    return counts


def avg_over_groups(v, g, n_groups):
    """
    Args:
        v (Tensor): Vector containing the quantity to average over.
        g (Tensor): Vector of the same length as v, containing group information.
    Returns:
        group_avgs (Tensor): Vector of length num_groups
        group_counts (Tensor)
    """
    assert v.device == g.device
    assert v.numel() == g.numel()
    group_count = get_counts(g, n_groups)
    group_sum = torch.zeros(n_groups, device=v.device)
    for i in range(n_groups):
        mask = (g == i)
        if mask.any():
            group_sum[i] = v[mask].sum()
    group_avgs = group_sum / group_count
    group_avgs[group_count == 0] = float('nan')
    return group_avgs, group_count


def map_to_id_array(df, ordered_map={}):
    maps = {}
    array = np.zeros(df.shape)
    for i, c in enumerate(df.columns):
        if c in ordered_map:
            category_type = CategoricalDtype(categories=ordered_map[c],
                                             ordered=True)
        else:
            category_type = 'category'
        series = df[c].astype(category_type)
        maps[c] = series.cat.categories.values
        array[:, i] = series.cat.codes.values
    return maps, array


def subsample_idxs(idxs, num=5000, take_rest=False, seed=None):
    seed = (seed + 541433) if seed is not None else None
    rng = np.random.default_rng(seed)
    idxs = idxs.copy()
    rng.shuffle(idxs)
    if take_rest:
        idxs = idxs[num:]
    else:
        idxs = idxs[:num]
    return idxs


def shuffle_arr(arr, seed=None):
    seed = (seed + 548207) if seed is not None else None
    rng = np.random.default_rng(seed)
    arr = arr.copy()
    rng.shuffle(arr)
    return arr


def threshold_at_recall(y_pred, y_true, global_recall=60):
    """Calculate the model threshold used to achieve a desired global_recall level.

    Args:
        y_pred (Description of y_pred, Assumes that y_true is a vector of the true binary labels.)
        y_true (Description of y_true.)
        global_recall (Description of global_recall.)
    """
    return np.percentile(y_pred[y_true == 1], 100 - global_recall)


def numel(obj):
    if torch.is_tensor(obj):
        return obj.numel()
    elif isinstance(obj, list):
        return len(obj)
    else:
        raise TypeError('Invalid type for numel')


def format_eval_results(results: Dict[str, Any], dataset) -> str:
    """Format evaluation results into well-formatted tables.

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
            elif 'source' in key and 'count' not in key:
                group_id = int(key.split(':')[1])
                source_data.append({
                    'Source': source_names[group_id],
                    'Score': value,
                    'Count': metric_results.get(f'count_source:{group_id}', 0)
                })

        # Create source table
        if source_data:
            # Sort by score descending
            source_data.sort(key=lambda x: x['Score'], reverse=True)

            formatted_output.append("\nSource-wise Results:")
            formatted_output.append("-" * 50)
            formatted_output.append(
                f"{'Source':<25} {'Score':<10} {'Count':<10}")
            formatted_output.append("-" * 50)

            for row in source_data:
                formatted_output.append(
                    f"{row['Source']:<25} {row['Score']:<10.3f} {row['Count']:<10}"
                )

        # Add summary statistics
        formatted_output.append("\nSummary Statistics:")
        formatted_output.append("-" * 40)
        if avg_score is not None and not np.isnan(avg_score):
            formatted_output.append(f"Average {metric_name}: {avg_score:.3f}")
        if worst_group_score is not None and not np.isnan(worst_group_score):
            formatted_output.append(
                f"Worst-group {metric_name}: {worst_group_score:.3f}")

        scores = [row['Score'] for row in source_data]
        if len(scores) > 0:
            formatted_output.append(f"Min {metric_name}: {min(scores):.3f}")
            formatted_output.append(f"Max {metric_name}: {max(scores):.3f}")
            formatted_output.append(f"Std {metric_name}: {np.std(scores):.3f}")

    return "\n".join(formatted_output)
