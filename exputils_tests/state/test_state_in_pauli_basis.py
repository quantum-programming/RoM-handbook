import qutip
import numpy as np
from itertools import product
from exputils.state.state_in_pauli_basis import _make_n_paulis, state_in_pauli_basis


def fast_trace(n: int, state: np.ndarray, pauli_str):
    # assert len(pauli_str) == n
    cols_table = [
        [0, 1],  # I      diagonal
        [1, 0],  # X anti-diagonal
        [1, 0],  # Y anti-diagonal
        [0, 1],  # Z      diagonal
    ]
    vals_table = [
        [0, 0],  # I [  1,  1]
        [0, 0],  # X [  1,  1]
        [3, 1],  # Y [-1j, 1j]
        [0, 2],  # Z [  1, -1]
    ]
    phase_to_val = [1, 1j, -1, -1j]  # e^(idx*iπ/2) = phase_to_val[idx]
    ans = 0j
    for row in range(2**n):
        col = 0
        val = 0
        for j in range(n):
            col += cols_table[pauli_str[j]][(row >> (n - 1 - j)) & 1] << (n - 1 - j)
            val += vals_table[pauli_str[j]][(row >> (n - 1 - j)) & 1]
        # Be careful that the order of (col, row) is reversed from the state.
        ans += state[col][row] * phase_to_val[val & 0b11]
    return ans


def state_in_pauli_basis_slow(state: np.ndarray):
    basis_count = state.shape[0]
    assert (
        state.shape[0] >= 2 and bin(state.shape[0]).count("1") == 1
    ), "state.shape[0] must be expressed as 2^n, where n is a positive integer."

    n_qubit = state.shape[0].bit_length() - 1
    assert 2**n_qubit == basis_count

    if state.ndim == 1:
        n_paulis = _make_n_paulis(n_qubit)
        ret = np.array([state.conj() @ pauli @ state for pauli in n_paulis])
        assert np.allclose(ret.imag, 0)
        return np.array(ret.real, dtype=np.float64)
    elif state.ndim == 2:
        # range(4) -> "IXYZ"
        ret = np.array(
            [
                fast_trace(n_qubit, state, pauli_tuple)
                for pauli_tuple in product(range(4), repeat=n_qubit)
            ]
        )
        n_paulis = _make_n_paulis(n_qubit)
        assert np.allclose(
            np.array([np.trace(state @ pauli) for pauli in n_paulis]).imag, 0
        ), "np.trace(state @ pauli) must be real."
        assert np.allclose(
            ret.real,
            np.array([np.trace(state @ pauli) for pauli in n_paulis]),
        ), "ret must be close to np.trace(state @ pauli)."
        assert np.allclose(ret.imag, 0), "ret must be real."
        return np.array(ret.real, dtype=np.float64)
    else:
        raise TypeError()


def test_state_in_pauli_basis():
    for n_qubit in range(1, 4 + 1):
        for _density in range(10 + 1):
            density = _density / 10.0
            dm = qutip.rand_dm(2**n_qubit, density=density).full()
            expected = state_in_pauli_basis_slow(dm)
            actual = state_in_pauli_basis(dm)
            assert np.allclose(expected, actual)
        print("ok!")
    print("all ok!")


if __name__ == "__main__":
    test_state_in_pauli_basis()
