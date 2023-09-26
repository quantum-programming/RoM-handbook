from typing import List, Tuple

import numpy as np
from numba import njit
from functools import lru_cache
from tqdm.auto import tqdm

from exputils.math.f2_irreducible_polynomial import enumerate_irreducible_polynomials
from exputils.stabilizer_group import generator_to_group_in_phase_pidx


@njit(cache=True)
def find_matrices_by_irreducible_polynomial(n: int, f: int):
    a_arr = np.empty(2 * n - 1, dtype=np.int_)
    a = 1
    for i in range(2 * n - 1):
        a_arr[i] = a
        a <<= 1
        if a > a ^ f:
            a ^= f
    symmetric_matrix_basis = np.empty((n, n, n), dtype=np.int_)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                symmetric_matrix_basis[i][j][k] = (a_arr[j + k] >> i) & 1
    matrices = np.zeros((1 << n, n, n), dtype=np.int_)
    for s in range(1 << n):
        for i in range(n):
            if s & (1 << i):
                matrices[s, :, :] ^= symmetric_matrix_basis[i, :, :]
    return matrices


@lru_cache
def cover_states(n: int) -> List[List[str]]:
    """finds minimum size (2^n+1) of "cover" states.

    The cover states satisfy the following condition:
    for any Pauli operator P,
    there exists a state of the cover states
    such that P stabilizes the state up to a phase.
    """
    f = enumerate_irreducible_polynomials(n)[n][0]
    z_matrices = find_matrices_by_irreducible_polynomial(n, f)

    states = []

    logical_0_list = [["I"] * n for _ in range(n)]
    for i in range(n):
        logical_0_list[i][i] = "Z"
    logical_0_str = ["".join(row) for row in logical_0_list]
    states.append(logical_0_str)

    diagonal_dict = {"I": "X", "Z": "Y"}
    for i in range(2**n):
        state_list = [["I"] * n for _ in range(n)]
        for j in range(n):
            for k in range(n):
                state_list[j][k] = "Z" if z_matrices[i][j][k] else "I"
        for j in range(n):
            state_list[j][j] = diagonal_dict[state_list[j][j]]
        state_str = ["".join(row) for row in state_list]
        states.append(state_str)
    return states


@lru_cache
def make_cover_info(n_qubit: int) -> Tuple[List[List[str]], np.ndarray, np.ndarray]:
    """Provide all information needed to use the cover state.
    Since this function is cached, you can call this function many times.

    Args:
        n_qubit (int): number of qubits

    Returns:
        Tuple[List[List[str]], np.ndarray, np.ndarray]: (cover_generators, cover_idxs, cover_vals)

    Examples:
        n_qubit=2
        cover_generators = [['ZI', 'IZ'], ['XI', 'IX'], ['YI', 'IY'], ['XZ', 'ZY'], ['YZ', 'ZX']]
        cover_idxs = array([[ 0(II), 12(ZI),  3(IZ), 15(ZZ)],
                            [ 0(II),  4(XI),  1(IX),  5(XX)],
                            [ 0(II),  8(YI),  2(IY), 10(YY)],
                            [ 0(II),  7(XZ), 14(ZY),  9(-YX)],
                            [ 0(II), 11(YZ), 13(ZX),  6(-XY)]])
        cover_vals = array([[ 0.2  ,  1.   ,  1.   ,  1.    ],
                            [ 0.2  ,  1.   ,  1.   ,  1.    ],
                            [ 0.2  ,  1.   ,  1.   ,  1.    ],
                            [ 0.2  ,  1.   ,  1.   , -1.    ],  (0.2*len(cover_generators) == 1.0)
                            [ 0.2  ,  1.   ,  1.   , -1.    ]]) (-1 for YX,XY)
    """
    cover_generators = cover_states(n_qubit)
    assert len(cover_generators) == 2**n_qubit + 1
    cover_idxs = np.zeros((len(cover_generators), 2**n_qubit), dtype=np.int32)
    cover_vals = np.zeros((len(cover_generators), 2**n_qubit), dtype=np.float64)
    seen_cnt = np.zeros(4**n_qubit, dtype=np.int32)
    for i, cover_generator in tqdm(
        enumerate(cover_generators),
        total=len(cover_generators),
        desc="make_cover_info",
        leave=False,
    ):
        g_phase, g_pidx = generator_to_group_in_phase_pidx(n_qubit, cover_generator)
        seen_cnt[g_pidx] += 1
        assert g_phase.shape == g_pidx.shape == (2**n_qubit,)
        assert np.sum(g_phase == 2) + np.sum(g_phase == 0) == 2**n_qubit
        assert np.all(0 <= g_pidx) and np.all(g_pidx < 4**n_qubit)
        cover_idxs[i] = g_pidx
        cover_vals[i] = 1 - g_phase  # exp^{0*πi/2}-> 1, exp^{2*πi/2} -> -1
    cover_vals[:, 0] = 1 / (2**n_qubit + 1)
    assert seen_cnt[0] == (2**n_qubit + 1) and np.allclose(seen_cnt[1:], 1)
    return (cover_generators, cover_idxs, cover_vals)


def main():
    for n_qubit in range(1, 10 + 1):
        cover_generators, cover_idxs, cover_vals = make_cover_info(n_qubit)
        print(f"{cover_idxs[:10]=}")
        print(f"{cover_vals[:10]=}")
        print(f"{cover_generators[:10]=}")


if __name__ == "__main__":
    main()
