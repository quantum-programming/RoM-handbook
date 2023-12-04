from typing import List, Tuple

import numpy as np
from numba import njit

from exputils.dot.pauli_dot import pauli_dot_without_str
from exputils.math.f2_linear_equation_solver import (
    gauss_jordan,
    solve_f2_linear_equation,
)


def pauli_str_to_tableau_row(pauli_str: str) -> Tuple[List[int], List[int]]:
    assert all(c in "IXYZ" for c in pauli_str)
    return (
        [int(c in "XY") for c in pauli_str],  # X side
        [int(c in "ZY") for c in pauli_str],  # Z side
    )


def pauli_strs_to_tableau(pauli_strs: List[str]) -> List[List[int]]:
    return [
        (lambda x: x[0] + x[1])(pauli_str_to_tableau_row(pauli_str))
        for pauli_str in pauli_strs
    ]


@njit(cache=True)
def idx_to_pauli_str(n: int, idx: int) -> str:
    assert 0 <= idx <= 4**n
    ret = ""
    while idx:
        ret += "IXYZ"[idx & 0b11]
        idx >>= 2
    return "I" * (n - len(ret)) + ret[::-1]


@njit(cache=True)
def pauli_str_to_idx(n: int, pauli_str: str) -> int:
    # assert all(c in "IXYZ" for c in pauli_str)
    assert len(pauli_str) == n
    idx = 0
    for c in pauli_str:
        idx *= 4
        idx += ["I", "X", "Y", "Z"].index(c)
    return idx


def tableau_row_to_pauli_str(tableau_row: List[int]) -> str:
    n = len(tableau_row) // 2
    return "".join("IXZY"[tableau_row[i] + 2 * tableau_row[n + i]] for i in range(n))


def tableau_to_pauli_strs(tableau: List[List[int]]) -> List[str]:
    n = len(tableau)
    assert all(len(tableau[i]) == 2 * n for i in range(n))
    assert all(tableau[i][j] in [0, 1] for i in range(n) for j in range(2 * n))
    return [tableau_row_to_pauli_str(tableau_row) for tableau_row in tableau]


def gen_span(n: int, _tableau: List[List[int]]) -> Tuple[int, ...]:
    span = set()
    tableau = [sum(val << idx for idx, val in enumerate(t)) for t in _tableau]
    for bit in range(1 << n):
        ha = 0
        for i in range(n):
            if (bit >> i) & 1:
                ha ^= tableau[i]
        span.add(ha)
    return tuple(sorted(list(span)))


def is_valid_tableau(tableau: List[List[int]]) -> bool:
    """check if tableau is valid stabilizer group

    Args:
        tableau (List[List[int]]): tableau rows concatenated (X,Z)

    Returns:
        bool: True if tableau is valid stabilizer group
    """
    n = len(tableau)
    assert len(tableau[0]) == 2 * n
    tableau_in_bit = [sum(val << idx for idx, val in enumerate(t)) for t in tableau]
    # commutativity
    for i in range(n):
        reversed_tableau_row_in_bit = sum(
            tableau[i] << (j + (-1 if j >= n else +1) * n) for j in range(2 * n)
        )
        for j in range(i + 1, n):
            if bin(reversed_tableau_row_in_bit & tableau_in_bit[j]).count("1") % 2 == 1:
                return False
    # dependence
    if gauss_jordan(tableau_in_bit, 2 * n, n)[0] != n:
        return False
    return True


def stabilizer_group_size_from_gen(generators: List[str]) -> int:
    n = len(generators[0])
    return (2**n) * len(generators)


def total_stabilizer_group_size(n: int) -> int:
    ret = 2**n
    for k in range(n):
        ret *= (2 ** (n - k)) + 1
    return ret


def get_valid(prev: List[str], bit: int) -> str:
    assert 1 <= bit < 2 ** len(prev)
    # bitのi桁目が
    # 0->prev[i]と可換 1->prev[i]と反可換
    n = len(prev)
    tableau = [t[n:] + t[:n] for t in pauli_strs_to_tableau(prev)]
    ok_cand = solve_f2_linear_equation(tableau, bit)
    assert all((np.dot(tableau[i], ok_cand)) % 2 == (bit >> i) & 1 for i in range(n))
    return tableau_row_to_pauli_str(ok_cand)


def _generator_to_phase_idx_format(
    generator: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    gens_phase = []
    gens_idx = []
    for gen_str in generator:
        if gen_str[0] == "-":
            gens_phase.append(2)
            gens_idx.append(["IXYZ".index(c) for c in gen_str[1:]])
        else:
            gens_phase.append(0)
            gens_idx.append(["IXYZ".index(c) for c in gen_str])
    return np.array(gens_phase, dtype=np.int32), np.array(gens_idx, dtype=np.int8)


@njit(cache=True)
def _generator_all(
    n_qubit: int, gens_phase: np.ndarray, gens_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    assert n_qubit == len(gens_idx) and n_qubit == len(gens_idx[0])
    group_phase = np.zeros(1 << n_qubit, dtype=np.int32)
    group_idx = np.zeros((1 << n_qubit, n_qubit), dtype=np.int8)
    for i in range(len(gens_phase)):
        gen_phase = gens_phase[n_qubit - 1 - i]
        gen_idx = gens_idx[n_qubit - 1 - i]
        for _j in range(1 << i):
            j = (1 << i) - 1 - _j
            g_phase = group_phase[j]
            g_idx = group_idx[j]
            group_phase[2 * j] = g_phase
            group_idx[2 * j] = g_idx
            dot_phase, dot_idx = pauli_dot_without_str(
                (g_phase, g_idx), (gen_phase, gen_idx)
            )
            group_phase[2 * j + 1] = dot_phase
            group_idx[2 * j + 1] = dot_idx
    return group_phase, group_idx


@njit(cache=True)
def convert_phase_idx_format_to_str_with_sign(
    group_phase: np.ndarray, group_idx: np.ndarray
) -> List[str]:
    ret = ["" for _ in range(group_phase.shape[0])]
    for i in range(group_phase.shape[0]):
        phase = group_phase[i]
        assert phase == 0 or phase == 2
        ret[i] = ("" if phase == 0 else "-") + "".join(
            ["IXYZ"[c] for c in group_idx[i]]
        )
    return ret


@njit(cache=True)
def idx_to_pidx(n_qubit: int, group_idx: np.ndarray) -> np.ndarray:
    group_pidx = np.zeros(1 << n_qubit, dtype=np.int32)
    for i in range(1 << n_qubit):
        for j in range(n_qubit):
            group_pidx[i] <<= 2
            group_pidx[i] += group_idx[i, j]
    return group_pidx


def generator_to_group(n_qubit: int, generator: List[str]) -> List[str]:
    gens_phase, gens_idx = _generator_to_phase_idx_format(generator)
    assert np.dtype(gens_phase.dtype) == np.int32
    assert np.dtype(gens_idx.dtype) == np.int8
    group_phase, group_idx = _generator_all(n_qubit, gens_phase, gens_idx)
    return convert_phase_idx_format_to_str_with_sign(group_phase, group_idx)


def generator_to_group_in_phase_pidx(
    n_qubit: int, generator: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    gens_phase, gens_idx = _generator_to_phase_idx_format(generator)
    assert np.dtype(gens_phase.dtype) == np.int32
    assert np.dtype(gens_idx.dtype) == np.int8
    group_phase, group_idx = _generator_all(n_qubit, gens_phase, gens_idx)
    group_pidx = idx_to_pidx(n_qubit, group_idx)  # np.int32
    return group_phase, group_pidx


def main():
    from exputils.actual_generators import get_actual_generators

    for n_qubit in range(1, 5 + 1):
        gen = get_actual_generators(n_qubit)[0]
        print(f"n_qubit={n_qubit}")
        print(f"generator={gen}")
        print(f"stabilizer_group={generator_to_group(n_qubit, gen)}")


if __name__ == "__main__":
    main()
