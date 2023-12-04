import pickle
from functools import lru_cache
from importlib.resources import files
from typing import Dict, List, Tuple

import numpy as np
from numba import njit
from scipy.sparse import csc_matrix, load_npz
from tqdm import tqdm


def get_H_in_H_basis(n: int) -> np.ndarray:
    H_state = []
    factorial_table = [1] * (n + 1)
    for k in range(n):
        factorial_table[k + 1] = factorial_table[k] * (k + 1)
    for i in range(n + 1):
        H_state.append(
            2 ** (i / 2)
            * factorial_table[n]
            / factorial_table[i]
            / factorial_table[n - i]
        )
    return np.array(H_state)


@lru_cache
def get_H_Amat(n: int) -> csc_matrix:
    return load_npz(files("exputils").joinpath(f"../data/LCgraphs/H_Amat_{n}.npz"))


def get_H_Amats(directory: str = "LCgraphs") -> Dict[int, np.ndarray]:
    actual_Amats = dict()
    try:
        sz = 1
        while True:
            actual_Amats[sz] = get_H_Amat(sz, directory)
            sz += 1
    except FileNotFoundError:
        pass
    return actual_Amats


@lru_cache
def get_group_to_H(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """It returns an ancillary data for generating H Amat.

    The ancillary data consists of two 2D arrays `idxs` and `data` and 1D array `unique_indices`.
    Each row of `idxs` and `data` corresponds to a matrix.
    The matrices can be represented in csc as `Ms[i] = csc_matrix((data[i], idxs[i], np.arange(len(data.shape[1]) + 1)))`.
    That is, each column of Ms[i] contains only one nonzero element,
    and as for j-th column idxs[i][j]-th element equals data[i][j].
    Then, `np.hstack([M @ H for M in Ms])[unique_indices]` corresponds to perm Amat columns, where H is a Walsh matrix.
    """
    loaded = np.load(files("exputils").joinpath(f"../data/LCgraphs/group_to_H_{n}.npz"))
    idxs = loaded["idxs"]
    data = loaded["data"]
    unique_indices = loaded["unique_indices"]
    return idxs, data, unique_indices
