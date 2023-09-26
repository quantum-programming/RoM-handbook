import pickle
from functools import lru_cache
from importlib.resources import files
from typing import Dict, List, Tuple

import numpy as np
from numba import njit
from scipy.sparse import csc_matrix, load_npz
from tqdm import tqdm


def get_row_size(n: int) -> int:
    return (n + 3) * (n + 2) * (n + 1) // 6


def get_row_info(n: int) -> List[Tuple[int, int, int]]:
    basis = []
    for x in range(n + 1):
        for y in range(n + 1):
            for z in range(n + 1):
                if x + y + z <= n:
                    basis.append((x, y, z))
    return basis


def tensor_product_in_perm_basis(state: np.ndarray, n: int) -> np.ndarray:
    perm_basis = get_row_info(n)
    perm_state = []
    factorial_table = [1] * (n + 1)
    for k in range(n):
        factorial_table[k + 1] = factorial_table[k] * (k + 1)
    for x, y, z in perm_basis:
        perm_state.append(
            state[1] ** x
            * state[2] ** y
            * state[3] ** z
            * factorial_table[n]
            / factorial_table[x]
            / factorial_table[y]
            / factorial_table[z]
            / factorial_table[n - x - y - z]
        )
    return np.array(perm_state)


@lru_cache
def get_col_info(n: int, directory: str = "LCgraphs") -> List[List[List[str]]]:
    col_file = files("exputils").joinpath(f"../data/{directory}/col_info_{n}.pkl")
    with open(col_file, "rb") as f:
        col_info = pickle.load(f)
    return col_info


@lru_cache
def get_fully_entangled_col_info(n: int) -> List[List[List[str]]]:
    col_file = files("exputils").joinpath(
        f"../data/LCgraphs/fully_entangled_col_info_{n}.pkl"
    )
    with open(col_file, "rb") as f:
        col_info = pickle.load(f)
    return col_info


# @lru_cache
def get_perm_Amat(n: int, directory: str = "LCgraphs") -> csc_matrix:
    return load_npz(
        files("exputils").joinpath(f"../data/{directory}/perm_Amat_{n}.npz")
    )


def get_perm_Amats(directory: str = "LCgraphs") -> Dict[int, np.ndarray]:
    actual_Amats = dict()
    try:
        sz = 1
        while True:
            actual_Amats[sz] = get_perm_Amat(sz, directory)
            sz += 1
    except FileNotFoundError:
        pass
    return actual_Amats


@lru_cache
def get_fully_entangled_perm_Amat(n: int) -> csc_matrix:
    return load_npz(
        files("exputils").joinpath(
            f"../data/LCgraphs/fully_entangled_perm_Amat_{n}.npz"
        )
    )


def get_fully_entangled_perm_Amats() -> Dict[int, np.ndarray]:
    actual_Amats = dict()
    try:
        sz = 1
        while True:
            actual_Amats[sz] = get_fully_entangled_perm_Amat(sz)
            sz += 1
    except FileNotFoundError:
        pass
    return actual_Amats


@lru_cache
def get_group_to_perm(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """It returns an ancillary data for generating perm Amat.

    The ancillary data consists of two 2D arrays `idxs` and `data` and 1D array `unique_indices`.
    Each row of `idxs` and `data` corresponds to a matrix.
    The matrices can be represented in csc as `Ms[i] = csc_matrix((data[i], idxs[i], np.arange(len(data.shape[1]) + 1)))`.
    That is, each column of Ms[i] contains only one nonzero element,
    and as for j-th column idxs[i][j]-th element equals data[i][j].
    Then, `np.hstack([M @ H for M in Ms])[unique_indices]` corresponds to perm Amat columns, where H is a Walsh matrix.
    """
    loaded = np.load(
        files("exputils").joinpath(f"../data/LCgraphs/group_to_perm_{n}.npz")
    )
    idxs = loaded["idxs"]
    data = loaded["data"]
    unique_indices = loaded["unique_indices"]
    return idxs, data, unique_indices


@njit(cache=True)
def _fwht_inplace(n, v):
    assert 1 << n == v.shape[0]
    h = 1
    for _ in range(n):
        for i in range(0, len(v), h << 1):
            for j in range(i, i + h):
                x = v[j]
                y = v[j + h]
                v[j] = x + y
                v[j + h] = x - y
        h <<= 1


def make_group_to_perm_unique(n, group_to_perm_idxs, group_to_perm_data):
    row_size = get_row_size(n)
    rng = np.random.default_rng(0)
    random_vector = rng.integers(-1e18, 1e18, row_size)

    set_of_hash = set()
    used_indices = []
    unique_indices = []
    for i in tqdm(range(group_to_perm_data.shape[0])):
        dots = random_vector[group_to_perm_idxs[i]] * group_to_perm_data[i]
        _fwht_inplace(n, dots)
        need = False
        # print(dots)
        for r, val in enumerate(dots):
            if val not in set_of_hash:
                set_of_hash.add(val)
                unique_indices.append((i << n) + r)
                need = True
        if need:
            used_indices.append(i)
    return np.array(used_indices), np.array(unique_indices)
