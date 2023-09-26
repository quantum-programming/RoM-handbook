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

# stabilizer_simulator 直下での実行を想定


def direct_sum(n_list: List[int], A_list: Iterable[List[int]]) -> List[int]:
    shift = 0
    ret = []
    for n, A in zip(n_list, A_list):
        for row in A:
            ret.append(row << shift)
        shift += n
    return ret


@lru_cache()
def get_connected_LCgraphs(n: int) -> List[List[int]]:
    LCgraph_file = f"data/LCgraphs/vncorbits{n}.npy"
    with open(LCgraph_file, "rb") as f:
        LCgraphs = np.load(f)
    return LCgraphs.tolist()


def enumerate_LCgraphs(n: int) -> List[List[int]]:
    union = []
    for n_list in partitions(n):
        iterators = [get_connected_LCgraphs(k) for k in n_list]
        for A_list in product(*iterators):
            union.append(direct_sum(n_list, A_list))
    return union


def LCgraph_orbit(n: int, A: List[int]) -> Set[Tuple[int, ...]]:
    eye = [1 << i for i in range(n)]
    XZ_tableau = eye + A
    iterators = []
    for i in range(n):
        it = []
        col_pairs = []
        for d in range(6):
            X = XZ_tableau[i]
            Z = XZ_tableau[i + n]
            if d % 3 == 1:
                Z ^= X
            if d % 3 == 2:
                X ^= Z
            if d % 2:
                X, Z = Z, X
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
                tableau[i + n] ^= tableau[i]
            if d % 3 == 2:
                tableau[i] ^= tableau[i + n]
            if d % 2:
                tableau[i], tableau[i + n] = tableau[i + n], tableau[i]
        orbit.add(tuple(tableau))
    return orbit


def graphs_to_tableaux(n: int, LCgraphs: List[List[int]]) -> Set[Tuple[int, ...]]:
    orbit_union = set()
    for LCgraph in LCgraphs:
        orbit_union |= LCgraph_orbit(n, LCgraph)
    return orbit_union


@njit("i1[:, :](i1[:, :])", cache=True)
def count_xyz_group_idx(group_idx: np.ndarray):
    """count 1, 2, 3 in the given 2d array.

    Args:
        group_idx (np.ndarray): an array which represents pauli operators.
    """
    row, col = group_idx.shape
    xyzs = np.zeros((3, row), np.int8)
    for i in range(row):
        for j in range(col):
            if group_idx[i][j]:
                xyzs[group_idx[i][j] - 1][i] += 1
    return xyzs


@njit(cache=True)
def add_at(size, indices, data):
    col_data = np.zeros(size, np.int16)
    for i in range(len(indices)):
        col_data[indices[i]] += data[i]
    return col_data


def make_group_to_perm(
    n: int, LCgraphs: List[List[int]]
) -> Tuple[List[List[str]], csc_matrix]:
    basis = get_row_info(n)
    basis_index_np = np.zeros((n + 1, n + 1, n + 1), np.int16)
    for i, e in enumerate(basis):
        basis_index_np[e] = i

    tableaux = graphs_to_tableaux(n, LCgraphs)

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
        idxs[i] = basis_index_np[xyzs[0], xyzs[1], xyzs[2]]
        data[i] = (1 - group_phase).astype(np.int16)

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

    file = f"data/LCgraphs/group_to_perm_{n}.npz"
    np.savez_compressed(file, idxs=idxs, data=data)


def make_perm_Amat_from_group_to_perm(n):
    idxs, data, unique_indices = get_group_to_perm(n)
    H = sylvesters(n)
    row_size = get_row_size(n)
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


def make_col_info_and_perm_Amat(
    n: int, LCgraphs: List[List[int]]
) -> Tuple[List[List[str]], csc_matrix]:
    basis = get_row_info(n)
    row_size = len(basis)
    basis_index_np = np.zeros((n + 1, n + 1, n + 1), np.int16)
    for i, e in enumerate(basis):
        basis_index_np[e] = i

    tableaux = graphs_to_tableaux(n, LCgraphs)

    H = list(sylvesters(n).astype(np.int16))
    signed_phase_array = [
        np.array(
            [((cidx >> i) & 1) << 1 for i in range(n)],
            dtype=np.int32,
        )
        for cidx in range(1 << n)
    ]
    gens_phase = np.zeros(n, dtype=np.int32)
    col_info = []
    csc_idxs_list = []
    csc_data_list = []
    hashed_columns = set()
    for tableau_tp in tqdm(tableaux):
        gens_idx = np.zeros((n, n), np.int8)
        x = np.array(tableau_tp[:n], np.int8)
        z = np.array(tableau_tp[n:], np.int8)
        for j in range(n):
            # get index of I, X, Y, Z from bit representation
            gens_idx[j] = ((x >> j) & 1) ^ (((z >> j) & 1) * 0b11)
        group_phase, group_idx = _generator_all(n, gens_phase, gens_idx)
        xyzs = count_xyz_group_idx(group_idx)
        to_basis_idxs = basis_index_np[xyzs[0], xyzs[1], xyzs[2]]
        unique_idxs, to_unique_idxs = np.unique(to_basis_idxs, return_inverse=True)
        phase_signs = (1 - group_phase).astype(np.int16)
        for cidx in range(1 << n):
            added_data = add_at(len(unique_idxs), to_unique_idxs, H[cidx] * phase_signs)
            nonzero_mask = added_data != 0
            col_idxs = unique_idxs[nonzero_mask]
            col_data = added_data[nonzero_mask]

            hs_col = tuple(col_data) + tuple(col_idxs)
            if hs_col in hashed_columns:
                continue
            hashed_columns.add(hs_col)

            csc_idxs_list.append(col_idxs)
            csc_data_list.append(col_data)

            signed_generator = convert_phase_idx_format_to_str_with_sign(
                signed_phase_array[cidx], gens_idx
            )
            col_info.append(signed_generator)

    col_size = len(csc_idxs_list)
    csc_indptr = [0] + list(accumulate(map(len, csc_idxs_list)))
    csc_idxs = np.concatenate(csc_idxs_list)
    csc_data = np.concatenate(csc_data_list)
    perm_Amat = csc_matrix(
        (csc_data, csc_idxs, csc_indptr), shape=(row_size, col_size), dtype=np.int16
    )
    return col_info, perm_Amat


def generate_fully_entangled_perm_Amat(n_max):
    for n in range(1, n_max + 1):
        LCgraphs = get_connected_LCgraphs(n)
        col_info, perm_Amat = make_col_info_and_perm_Amat(n, LCgraphs)

        col_info_file = f"data/LCgraphs/fully_entangled_col_info_{n}.pkl"
        with open(col_info_file, "wb") as f:
            pickle.dump(col_info, f)

        perm_Amat_file = f"data/LCgraphs/fully_entangled_perm_Amat_{n}.npz"
        save_npz(perm_Amat_file, perm_Amat)

        print(f"{n = }, {perm_Amat.shape = }")


def generate_perm_Amat(n_max):
    for n in range(1, n_max + 1):
        LCgraphs = enumerate_LCgraphs(n)
        col_info, perm_Amat = make_col_info_and_perm_Amat(n, LCgraphs)

        col_info_file = f"data/LCgraphs/col_info_{n}.pkl"
        with open(col_info_file, "wb") as f:
            pickle.dump(col_info, f)

        perm_Amat_file = f"data/LCgraphs/perm_Amat_{n}.npz"
        save_npz(perm_Amat_file, perm_Amat)

        print(f"{n = }, {perm_Amat.shape = }")


def main():
    for n in range(1, 7 + 1):
        perm_Amat = make_perm_Amat_from_group_to_perm(n)
        perm_Amat_file = f"data/LCgraphs/perm_Amat_{n}.npz"
        save_npz(perm_Amat_file, perm_Amat)
        print(f"{n = }, {perm_Amat.shape = }")


if __name__ == "__main__":
    main()
