from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def shuffled_group_kfold_split(
    groups,
    n_splits: int,
    random_state: int | None = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
   
    groups = np.asarray(groups).astype(str)
    unique_groups, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    n_groups = len(unique_groups)
    if n_groups < 2:
        raise ValueError("At least two unique groups are required for grouped cross-validation.")
    n_splits = int(min(max(2, n_splits), n_groups))

    rng = np.random.default_rng(random_state)
    
    jitter = rng.random(n_groups)
    order = np.lexsort((jitter, -counts))

    fold_sizes = np.zeros(n_splits, dtype=int)
    group_to_fold = np.empty(n_groups, dtype=int)
    for group_idx in order:
        min_size = fold_sizes.min()
        candidates = np.flatnonzero(fold_sizes == min_size)
        fold = int(rng.choice(candidates))
        group_to_fold[group_idx] = fold
        fold_sizes[fold] += int(counts[group_idx])

    row_folds = group_to_fold[inverse]
    all_idx = np.arange(len(groups))
    for fold in range(n_splits):
        valid_idx = all_idx[row_folds == fold]
        train_idx = all_idx[row_folds != fold]
        if len(valid_idx) == 0 or len(train_idx) == 0:
            raise RuntimeError("Generated an empty grouped CV fold; reduce n_splits.")
        yield train_idx, valid_idx
