from functools import reduce

import numpy as np

from exputils.state.random import (
    _make_random_mixed_density_matrix,
    _make_random_pure_density_matrix,
)
from exputils.state.state_in_pauli_basis import state_in_pauli_basis


def make_random_tensor_product_state(
    kind: str, n_qubit: int, seed: int = 0, check: bool = False
) -> np.ndarray:
    assert kind in ["H", "F", "W", "mixed", "pure"], kind
    if kind == "H":
        # "Robustness of Magic and Symmetries of the Stabiliser Polytope"
        # |H〉 := T |+〉 = 1/√2 (|0〉 + e^{iπ/4} |1〉)
        H_state = np.array([(1.0 + 0.0j) / np.sqrt(2), 0.5 + 0.5j], dtype=np.complex_)
        states = [H_state.copy()] * n_qubit
    elif kind == "F":
        # Reference: https://en.wikipedia.org/wiki/Magic_state_distillation
        # |F〉:= cos(β/2) |0〉+ e^{iπ/4} sin(β/2) |1〉, where β = arccos(1/√3)
        beta = np.arccos(1 / np.sqrt(3))
        F_state = np.array(
            [np.cos(beta / 2), np.exp(1j * np.pi / 4) * np.sin(beta / 2)],
            dtype=np.complex_,
        )
        states = [F_state.copy()] * n_qubit
    elif kind == "W":
        # Reference: https://en.wikipedia.org/wiki/W_state
        # |W〉:= 1/√3 (|001〉 + |010〉 + |100〉)
        W_state = np.zeros(2**n_qubit, dtype=np.complex_)
        for i in range(n_qubit):
            W_state[2**i - 1] = 1 / np.sqrt(n_qubit)
        states = [W_state.copy()]
    elif kind == "mixed":
        assert 0 <= seed < 1000, f"seed must be in [0, 1000), but {seed=}"
        states = []
        for i in range(n_qubit):
            dm = _make_random_mixed_density_matrix(1, seed * 1000 + i)
            states.append(dm)
    elif kind == "pure":
        assert 0 <= seed < 1000, f"seed must be in [0, 1000), but {seed=}"
        states = []
        for i in range(n_qubit):
            dm = _make_random_pure_density_matrix(1, seed * 1000 + i)
            states.append(dm)
    else:
        raise ValueError(f"Invalid kind: {kind}")

    states_in_pauli_basis = list(map(state_in_pauli_basis, states))
    ret = reduce(np.kron, states_in_pauli_basis)

    if check:
        assert np.allclose(ret, state_in_pauli_basis(reduce(np.kron, states), True))

    return ret
