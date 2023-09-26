from typing import List

import numpy as np
from scipy.sparse import csc_matrix

from exputils.actual_Amat import get_actual_Amat
from exputils.dot.load_data import load_dot_data
from exputils.math.fwht import sylvesters
from exputils.stabilizer_group import (
    generator_to_group,
    get_valid,
    pauli_str_to_idx,
    total_stabilizer_group_size,
)


def get_random_next_generator(n_qubit: int, prev: List[str]) -> List[str]:
    n_qubit += 1
    i = np.random.randint(2 * ((2 ** (n_qubit - 1)) - 1) + 3)
    if i >= 2 * ((2 ** (n_qubit - 1)) - 1):
        # Enumeration of commutative generators
        i -= 2 * ((2 ** (n_qubit - 1)) - 1)
        assert 0 <= i < 3
        return sorted(
            list(map(lambda row: row + "I", prev)) + ["I" * (n_qubit - 1) + "XYZ"[i]]
        )
    else:
        # Enumeration of anti-commutative generators
        bit = 1 + i // 2
        XorZ = i % 2
        head = [prev[i] + "IY"[(bit >> i) & 1] for i in range(n_qubit - 1)]
        anti_commute = get_valid(prev, bit)
        return sorted(head + [anti_commute + "XZ"[XorZ]])


def get_random_generator(n_qubit: int) -> List[str]:
    gen = "XYZ"[np.random.randint(3)]
    for i in range(1, n_qubit):
        gen = get_random_next_generator(i, gen)
    return gen


def _make_random_Amat_actual(n_qubit: int, sz: int) -> csc_matrix:
    actual_Amat = get_actual_Amat(n_qubit)
    idxs = np.random.choice(range(actual_Amat.shape[1]), sz, replace=False).astype(int)
    return actual_Amat[:, idxs]


def _make_random_Amat_dot_data(n_qubit: int, sz: int) -> csc_matrix:
    H = sylvesters(n_qubit)
    data_per_col, rows_per_col = load_dot_data(n_qubit)
    idxs = np.random.choice(range(data_per_col.shape[0]), sz, replace=True).astype(int)
    cols = np.repeat(np.arange(sz), 2**n_qubit)
    rows = rows_per_col[idxs].flatten()
    data = data_per_col[idxs].flatten() * H[
        np.random.randint(0, 2**n_qubit, sz)
    ].flatten().astype(np.int8)
    return csc_matrix((data, (rows, cols)), shape=(4**n_qubit, sz), dtype=np.int8)


def _make_random_Amat_gen(n_qubit: int, sz: int) -> csc_matrix:
    data = []
    rows = []
    cols = np.repeat(np.arange(sz), 2**n_qubit)
    for _ in range(sz):
        gen = get_random_generator(n_qubit)
        sign_bit = np.random.randint(1 << n_qubit)
        group = generator_to_group(n_qubit, gen)
        for idx, g in enumerate(group):
            isMinus = g[0] == "-"
            data.append(
                (1 if bin(sign_bit & idx)[2:].count("1") & 1 == 0 else -1)
                * (-1 if isMinus else 1)
            )
            rows.append(pauli_str_to_idx(n_qubit, g[1:] if isMinus else g))
    return csc_matrix((data, (rows, cols)), shape=(4**n_qubit, sz), dtype=np.int8)


def make_random_Amat(n_qubit: int, sz: int, method: str = None) -> csc_matrix:
    assert sz <= total_stabilizer_group_size(n_qubit)

    if method != None:
        if method == "actual":
            return _make_random_Amat_actual(n_qubit, sz)
        elif method == "dot_data":
            return _make_random_Amat_dot_data(n_qubit, sz)
        elif method == "gen":
            return _make_random_Amat_gen(n_qubit, sz)
        else:
            raise ValueError("method must be 'actual', 'dot_data', or 'gen'.")

    if n_qubit <= 5:
        Amat = _make_random_Amat_actual(n_qubit, sz)
    elif n_qubit == 6:
        Amat = _make_random_Amat_dot_data(n_qubit, sz)
    else:
        Amat = _make_random_Amat_gen(n_qubit, sz)

    return Amat
