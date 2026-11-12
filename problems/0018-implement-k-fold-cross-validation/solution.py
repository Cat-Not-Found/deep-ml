import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Return train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds
        shuffle: Whether to shuffle indices before splitting
    
    Returns:
        List of (train_indices, test_indices) tuples, where each is a list of ints
    """
    # Your implementation here
    indices = list(range(n_samples))
    if shuffle:
        random.shuffle(indices)

    fold_sizes = [n_samples // k] * k
    fold_sizes[0] += (n_samples % k)

    splits = []
    start = 0

    for fold_size in fold_sizes:

        # Itself as test set
        test_indices = indices[start:start + fold_size]

        # Others as validation set
        train_indices = indices[:start] + indices[start + fold_size:]
        splits.append((train_indices, test_indices))
        start += fold_size

    return splits
