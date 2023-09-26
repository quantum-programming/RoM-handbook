from typing import Union

import numpy as np
from numba import njit

from exputils.perm_Amat import get_group_to_perm, get_row_size


@njit(cache=True)
def _fwht_inplace_matrix(n, M):
    H, W = M.shape
    assert 1 << n == H
    h = 1
    for _ in range(n):
        for i in range(0, H, h << 1):
            for j in range(i, i + h):
                for k in range(W):
                    x = M[j][k]
                    y = M[j + h][k]
                    M[j][k] = x + y
                    M[j + h][k] = x - y
        h <<= 1


def compute_all_dot_products_perm(
    n: int,
    vec: np.ndarray,
    group_to_perm_idxs: Union[np.ndarray, None] = None,
    group_to_perm_data: Union[np.ndarray, None] = None,
    group_to_perm_unique_indices: Union[np.ndarray, None] = None,
) -> np.ndarray:
    vec = vec.flatten()
    assert get_row_size(n) == vec.shape[0]
    if (
        group_to_perm_idxs is None
        or group_to_perm_data is None
        or group_to_perm_unique_indices is None
    ):
        (
            group_to_perm_idxs,
            group_to_perm_data,
            group_to_perm_unique_indices,
        ) = get_group_to_perm(n)
    assert group_to_perm_idxs.shape == group_to_perm_data.shape
    answer_T = (vec[group_to_perm_idxs] * group_to_perm_data).T
    _fwht_inplace_matrix(n, answer_T)
    return answer_T.T.flatten()[group_to_perm_unique_indices]
