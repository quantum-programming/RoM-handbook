import os
import time
from itertools import combinations
from typing import Tuple

import numpy as np
from numba import njit
from tqdm.auto import tqdm

from exputils.math.q_binom import q_binomial
from exputils.stabilizer_group import (
    generator_to_group_in_phase_pidx,
    total_stabilizer_group_size,
)


@njit(cache=True)
def enumerate_RREF_helper(
    default_matrix: np.ndarray,
    not_decided_rows: np.ndarray,
    not_decided_cols: np.ndarray,
) -> np.ndarray:
    ret = np.zeros(
        (1 << not_decided_cols.shape[0], default_matrix.shape[0]), dtype=np.int32
    )
    sz = not_decided_cols.shape[0]
    for bit in range(1 << sz):
        matrix = default_matrix.copy()
        for i in range(sz):
            if bit & (1 << i):
                matrix[not_decided_rows[i]] += 1 << not_decided_cols[i]
        ret[bit] = matrix
    return ret


def enumerate_RREF(n: int, k: int):
    """enumerate all k times n RREF matrixes (row full rank)

    Reference:
        https://mathlandscape.com/rref-matrix
    """
    assert 1 <= n and 0 <= k <= n
    cnt = 0
    for _col_idxs_tuple in combinations(range(n), k):
        col_idxs_list = list(_col_idxs_tuple)  # which columns are (0, ..., 1, ..., 0)^T
        not_decided_rows = []  # which element can be 1 (row)
        not_decided_cols = []  # which element can be 1 (col)
        col_idxs_set = set(col_idxs_list)
        for row in range(k):
            for col in range(col_idxs_list[row] + 1, n):
                if col not in col_idxs_set:
                    not_decided_rows.append(row)
                    not_decided_cols.append(col)
        default_matrix = [0 for _ in range(k)]  # k times n matrix
        for row in range(k):
            default_matrix[row] += 1 << col_idxs_list[row]
        # enumerate 2^(len(not_decided_cols)) patterns
        matrixes = enumerate_RREF_helper(
            np.array(default_matrix, dtype=np.int32),
            np.array(not_decided_rows, dtype=np.int32),
            np.array(not_decided_cols, dtype=np.int32),
        )
        yield (col_idxs_list, col_idxs_set, matrixes)
        cnt += len(matrixes)
    assert cnt == q_binomial(n, k)


@njit(cache=True)
def enumerate_stabilizer_group_helper(
    default_matrix: np.ndarray,
    sym_diag_rows: np.ndarray,
    sym_diag_cols: np.ndarray,
    sym_non_diag_rows1: np.ndarray,
    sym_non_diag_cols1: np.ndarray,
    sym_non_diag_rows2: np.ndarray,
    sym_non_diag_cols2: np.ndarray,
) -> np.ndarray:
    for bit1 in range(1 << sym_diag_cols.shape[0]):
        matrix = default_matrix.copy()
        for i in range(sym_diag_cols.shape[0]):
            if bit1 & (1 << i):
                matrix[sym_diag_rows[i]] += 1 << sym_diag_cols[i]
        for bit2 in range(1 << sym_non_diag_cols1.shape[0]):
            matrix2 = matrix.copy()
            for i in range(sym_non_diag_cols1.shape[0]):
                if bit2 & (1 << i):
                    matrix2[sym_non_diag_rows1[i]] += 1 << sym_non_diag_cols1[i]
                    matrix2[sym_non_diag_rows2[i]] += 1 << sym_non_diag_cols2[i]
            yield matrix2


@njit(cache=True)
def convert_rref_to_default_matrix(
    rref: np.ndarray,
    n: int,
    k: int,
    col_idxs_list1: np.ndarray,
    col_idxs_list2: np.ndarray,
) -> np.ndarray:
    default_matrix = np.zeros((n), dtype=np.int32)  # n times 2n matrix
    # rref
    for row in range(k):
        default_matrix[row] += rref[row]
    # transpose from X side to Z side
    for row in range(k):
        for row2 in range(k, n):
            col = col_idxs_list2[row2 - k]
            if rref[row] & (1 << col):
                default_matrix[row2] += 1 << (n + col_idxs_list1[row])
    # Z side identity
    for row2 in range(k, n):
        default_matrix[row2] += 1 << (n + col_idxs_list2[row2 - k])
    return default_matrix


def enumerate_stabilizer_group(n: int):
    """enumerate all matrix representation of the stabilizer groups"""
    assert 1 <= n
    cnt = 0
    for k in range(n + 1):  # k is the rank of RREF matrix
        for col_idxs_list1, col_idxs_set, RREFs in enumerate_RREF(n, k):
            col_idxs_list2 = [i for i in range(n) if i not in col_idxs_set]
            assert len(col_idxs_list1) == k and len(col_idxs_list2) == n - k

            sym_diag_rows = np.array([k for k in range(k)], dtype=np.int32)
            sym_diag_cols = np.array(
                [n + col for col in col_idxs_list1], dtype=np.int32
            )
            sym_non_diag_rows1 = np.array(
                sum([[i] * (k - 1 - i) for i in range(k)], []), dtype=np.int32
            )
            sym_non_diag_cols1 = np.array(
                [n + c for c in sum([col_idxs_list1[i + 1 :] for i in range(k)], [])],
                dtype=np.int32,
            )
            sym_non_diag_rows2 = np.array(
                sum([list(range(k))[i + 1 :] for i in range(k)], []), dtype=np.int32
            )
            sym_non_diag_cols2 = np.array(
                sum([[n + c] * (k - 1 - i) for i, c in enumerate(col_idxs_list1)], []),
                dtype=np.int32,
            )
            assert len(sym_diag_rows) + len(sym_non_diag_rows1) == k * (k + 1) // 2

            for rref in RREFs:
                default_matrix = convert_rref_to_default_matrix(
                    rref,
                    n,
                    k,
                    np.array(col_idxs_list1, dtype=np.int32),
                    np.array(col_idxs_list2, dtype=np.int32),
                )

                # if True:
                #     print("-" * 5 + "default_matrix" + "-" * 5)
                #     print(
                #         *[
                #             ("0" * (2 * n - len(bin(row)[2:])) + bin(row)[2:])[::-1]
                #             for row in default_matrix
                #         ],
                #         sep="\n",
                #     )

                # add symmetric matrix to default_matrix (2**(k*(k*1)//2) patterns)
                for matrix in enumerate_stabilizer_group_helper(
                    default_matrix,
                    sym_diag_rows,
                    sym_diag_cols,
                    sym_non_diag_rows1,
                    sym_non_diag_cols1,
                    sym_non_diag_rows2,
                    sym_non_diag_cols2,
                ):
                    yield matrix
                    cnt += 1

                # if True:
                #     for matrix in matrixes:
                #         print("-" * 5)
                #         print(
                #             *[
                #                 ("0" * (2 * n - len(bin(row)[2:])) + bin(row)[2:])[::-1]
                #                 for row in matrix
                #             ],
                #             sep="\n",
                #         )
                #     print()

    assert cnt == total_stabilizer_group_size(n) // (2**n)


@njit(cache=True)
def generators_to_one_col_of_gen_data(n_qubit: int, matrix: np.ndarray):
    """
    Example:
        >>> generators_to_one_col_of_gen_data(3, np.array([0b110011, 0b001010, 0b000110]))
        (
            [the following three pauli operators are generator of the stabilizer group]

            0b110011 = (X:110, Z:011) = XYZ
            0b001010 = (X:010, Z:100) = ZXI
            0b000000 = (X:011, Z:000) = IXX

            [generate all elements of the stabilizer group by multiplying the generators]

            idx  res  idx  relationship  phase
            --- ----- --- -------------- -----
            000  +III   0                  1
          * 100  +XYZ  27                  1
          * 010  +ZXI  52                  1
            110  -YZZ  47 = 27 ^ 52       -1
          * 001  +IXX   5                  1
            101  +XZY  30 = 27      ^ 5    1
            011  +ZIX  49 =      52 ^ 5    1
            111  +YYY  42 = 27 ^ 52 ^ 5    1

            [given the information about which idx is minus, and generator idx,
             we can generate all elements of the stabilizer group speedily]
        )
        (array([False, False, False,  True, False, False, False, False]),
         array([27, 52,  5], dtype=uint16))
    """
    assert matrix.shape == (n_qubit,)
    assert n_qubit <= 7
    group_row = np.zeros(1 << n_qubit, dtype=np.uint16)
    group_phase = np.zeros(1 << n_qubit, dtype=np.int8)
    gen_idx = np.zeros(n_qubit, dtype=np.uint16)
    # make all matrix representation of the stabilizer group
    for i in range(n_qubit):
        for bit in range(1 << i):
            group_row[bit + (1 << i)] = group_row[bit] ^ matrix[i]
    # make phase (in modular 4, +0 means 0, +1 means +i, +2 means -1, +3 means -i)
    for i in range(n_qubit):
        for col in range(n_qubit):
            isX = (matrix[i] >> col) & 1
            isZ = (matrix[i] >> (n_qubit + col)) & 1
            if (not isX) and (not isZ):
                continue
            for bit in range(1 << i):
                isX2 = (group_row[bit] >> col) & 1
                isZ2 = (group_row[bit] >> (n_qubit + col)) & 1
                if isX and isZ:  # Y
                    if isX2 and (not isZ2):  # YX = -iZ
                        group_phase[bit + (1 << i)] += 3
                    elif (not isX2) and isZ2:  # YZ = iX
                        group_phase[bit + (1 << i)] += 1
                elif isX:  # X
                    if isX2 and isZ2:  # XY = iZ
                        group_phase[bit + (1 << i)] += 1
                    elif (not isX2) and isZ2:  # XZ = -iY
                        group_phase[bit + (1 << i)] += 3
                elif isZ:  # Z
                    if isX2 and isZ2:  # ZY = -iX
                        group_phase[bit + (1 << i)] += 3
                    elif isX2 and (not isZ2):  # ZX = iY
                        group_phase[bit + (1 << i)] += 1
    # make phase (convert to +1 or -1)
    for i in range(n_qubit):
        for bit in range(1 << i):
            group_phase[bit + (1 << i)] += group_phase[bit]
            group_phase[bit + (1 << i)] %= 4
    assert np.all((group_phase == 0) + (group_phase == 2) == 1)
    is_minus = group_phase == 2
    # make idx
    for i in range(n_qubit):
        for col in range(n_qubit):
            isX = (matrix[i] >> col) & 1
            isZ = (matrix[i] >> (n_qubit + col)) & 1
            if isX and isZ:
                gen_idx[i] += 2 << (2 * (n_qubit - 1 - col))
            elif isX:
                gen_idx[i] += 1 << (2 * (n_qubit - 1 - col))
            elif isZ:
                gen_idx[i] += 3 << (2 * (n_qubit - 1 - col))
    return is_minus, gen_idx


@njit(cache=True)
def make_gen_data_helper(
    i: int,
    ism_per_col: np.ndarray,
    gen_per_col: np.ndarray,
    is_minus: np.ndarray,
    gen_idx: np.ndarray,
):
    ism_per_col[i] = is_minus
    gen_per_col[i] = gen_idx


def generate_gen_data(n_qubit: int, CHUNK_CNT: int) -> Tuple[np.ndarray, np.ndarray]:
    """generate the representation of the Amat (ism_per_col, gen_per_col)

    Args:
        n_qubit (int): the number of qubits

    Returns:
        Tuple[np.ndarray, np.ndarray]: (ism_per_col, gen_per_col)
    """
    assert np.iinfo(np.uint16).min <= 0 and 4**n_qubit <= np.iinfo(np.uint16).max
    sz = total_stabilizer_group_size(n_qubit) // (2**n_qubit)
    assert sz % CHUNK_CNT == 0
    iterator = enumerate_stabilizer_group(n_qubit)
    for step in range(CHUNK_CNT):
        print(f"{step=} / {CHUNK_CNT=}")
        ism_per_col = np.zeros((sz // CHUNK_CNT, 2**n_qubit), dtype=np.bool_)
        gen_per_col = np.zeros((sz // CHUNK_CNT, n_qubit), dtype=np.uint16)
        for i in tqdm(range(sz // CHUNK_CNT)):
            matrix = next(iterator)
            is_minus, gen_idx = generators_to_one_col_of_gen_data(n_qubit, matrix)
            make_gen_data_helper(i, ism_per_col, gen_per_col, is_minus, gen_idx)
            # # check
            #
            # phase = np.ones(1 << n_qubit, dtype=np.int8)
            # phase[ism_per_col[i]] = -1
            # idx = gen_idxs_to_all_group_idxs(n_qubit, gen_idx)
            # generators = [
            #     "".join(
            #         "IXZY"[
            #             int((row >> (n_qubit + col)) & 1) * 2 + int((row >> col) & 1)
            #         ]
            #         for col in range(n_qubit)
            #     )
            #     for row in matrix
            # ]
            # _phase, _idx = generator_to_group_in_phase_pidx(n_qubit, generators)
            # _phase = 1 - _phase
            # assert np.all(phase == _phase)
            # assert np.all(idx == _idx)
        yield ism_per_col, gen_per_col
    assert next(iterator, None) is None


def make_gen_data(n_qubit: int, CHUNK_CNT: int):
    path = os.path.join(os.path.dirname(__file__), "../../data")
    assert os.path.exists(path)

    if not os.path.exists(path + "/genData"):
        os.mkdir(path + "/genData")

    for i, (ism_per_col, gen_per_col) in enumerate(
        generate_gen_data(n_qubit, CHUNK_CNT)
    ):
        path_for_npz = f"{path}/genData/{n_qubit}_{i}.npz"
        np.savez_compressed(
            path_for_npz, ism_per_col=ism_per_col, gen_per_col=gen_per_col
        )


if __name__ == "__main__":
    for n_qubit in range(1, 7 + 1):
        make_gen_data(n_qubit, 3 if n_qubit <= 6 else 225)
