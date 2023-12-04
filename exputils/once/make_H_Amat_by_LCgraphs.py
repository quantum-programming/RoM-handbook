import pickle
from functools import lru_cache
from itertools import accumulate, product
from typing import Iterable, List, Set, Tuple

import numpy as np
from numba import njit
from scipy.sparse import csc_matrix, hstack, save_npz
from tqdm import tqdm

from exputils.math.fwht import sylvesters
from exputils.math.partitions import partitions
from exputils.math.popcount import popcount
from exputils.perm_Amat import get_group_to_perm, get_row_info, get_row_size
from exputils.stabilizer_group import (
    _generator_all,
    convert_phase_idx_format_to_str_with_sign,
)
from exputils.once.make_perm_Amat_by_LCgraphs import (
    count_xyz_group_idx,
    add_at,
    enumerate_LCgraphs,
)
from exputils.H_Amat import get_group_to_H

# stabilizer_simulator 直下での実行を想定


def LCgraph_orbit_H(n: int, A: List[int]) -> Set[Tuple[int, ...]]:
    eye = [1 << i for i in range(n)]
    XZ_tableau = eye + A
    iterators = []
    for i in range(n):
        it = []
        col_pairs = []
        for d in range(3):
            X = XZ_tableau[i]
            Z = XZ_tableau[i + n]
            if d % 3 == 1:
                X, Z = Z, X
            if d % 3 == 2:
                X, Z = Z ^ X, X
            pair = (X, Z)
            if pair not in col_pairs:
                it.append(d)
                col_pairs.append(pair)
        iterators.append(it)

    orbit = set()
    for p in product(*iterators):
        tableau = XZ_tableau.copy()
        for i, d in enumerate(p):
            if d % 3 == 1:
                tableau[i], tableau[i + n] = tableau[i + n], tableau[i]
            if d % 3 == 2:
                tableau[i], tableau[i + n] = tableau[i + n] ^ tableau[i], tableau[i]
        orbit.add(tuple(tableau))
    return orbit


def graphs_to_tableaux_H(n: int, LCgraphs: List[List[int]]) -> Set[Tuple[int, ...]]:
    orbit_union = set()
    for LCgraph in LCgraphs:
        orbit_union |= LCgraph_orbit_H(n, LCgraph)
    return orbit_union


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


def make_group_to_H_unique(n, group_to_perm_idxs, group_to_perm_data):
    row_size = n + 1
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


def make_group_to_H(
    n: int, LCgraphs: List[List[int]]
) -> Tuple[List[List[str]], csc_matrix]:
    tableaux = graphs_to_tableaux_H(n, LCgraphs)

    gens_phase = np.zeros(n, dtype=np.int32)
    idxs = np.empty((len(tableaux), 2**n), np.int16)
    data = np.empty((len(tableaux), 2**n), np.int16)
    for i, tableau_tp in enumerate(tqdm(tableaux)):
        gens_idx = np.zeros((n, n), np.int8)
        x = np.array(tableau_tp[:n], np.int8)
        z = np.array(tableau_tp[n:], np.int8)
        for j in range(n):
            # get index of I, X, Y, Z from bit representation
            gens_idx[j] = ((x >> j) & 1) ^ (((z >> j) & 1) * 0b11)
        group_phase, group_idx = _generator_all(n, gens_phase, gens_idx)
        xyzs = count_xyz_group_idx(group_idx)
        idxs[i] = xyzs[0] + xyzs[1]
        data[i] = (1 - group_phase).astype(np.int16)
        data[i, xyzs[2] != 0] = 0

    rng = np.random.default_rng(0)
    hash_size = 4

    random_matrix_1 = rng.integers(
        -(2**15), 2**15 - 1, (2**n, hash_size), dtype=np.int16
    )
    random_matrix_2 = rng.integers(
        -(2**15), 2**15 - 1, (2**n, hash_size), dtype=np.int16
    )
    hashes = (idxs) @ random_matrix_1 + (data) @ random_matrix_2

    _, indices = np.unique(hashes, return_index=True, axis=0)
    idxs = idxs[indices]
    data = data[indices]

    used_indices, _ = make_group_to_H_unique(n, idxs, data)
    new_idxs = idxs[used_indices]
    new_data = data[used_indices]
    _, unique_indices = make_group_to_H_unique(n, new_idxs, new_data)

    file = f"data/LCgraphs/group_to_H_{n}.npz"
    np.savez_compressed(
        file, idxs=new_idxs, data=new_data, unique_indices=unique_indices
    )


def make_H_Amat_from_group_to_H(n):
    idxs, data, unique_indices = get_group_to_H(n)
    H = sylvesters(n)
    row_size = n + 1
    blocks = []
    iter_idx = 0
    mask = (1 << n) - 1
    for i in tqdm(range(idxs.shape[0])):
        block = np.zeros((row_size, 2**n), np.int16)
        for j in range(2**n):
            block[idxs[i][j]] += data[i][j] * H[j]
        begin_idx = iter_idx
        while iter_idx < len(unique_indices) and unique_indices[iter_idx] < (
            (i + 1) << n
        ):
            iter_idx += 1
        end_idx = iter_idx
        block_indices = unique_indices[begin_idx:end_idx] & mask
        blocks.append(csc_matrix(block)[:, block_indices])
    Amat = hstack(blocks)
    return Amat


def main():
    for n in range(1, 10 + 1):
        LCgraphs = enumerate_LCgraphs(n)
        make_group_to_H(n, LCgraphs)
        H_Amat = make_H_Amat_from_group_to_H(n)
        H_Amat_file = f"data/LCgraphs/H_Amat_{n}.npz"
        save_npz(H_Amat_file, H_Amat)
        print(f"{n = }, {H_Amat.shape = }")


if __name__ == "__main__":
    main()
