import torch
import numpy as np
from torch.utils.data import Subset
from pandas.api.types import CategoricalDtype


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
