from functools import reduce
from itertools import product
from typing import List

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix


def _make_n_paulis(n_qubit: int) -> List[csr_matrix]:
    identity = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    paulis = [identity, sigma_x, sigma_y, sigma_z]
    n_paulis = [
        csr_matrix(reduce(np.kron, pauli_tuple))
        for pauli_tuple in product(paulis, repeat=n_qubit)
    ]
    return n_paulis


@njit(cache=True)
def _state_in_pauli_basis_inplace_calculation(
    n_qubit: int, density_matrix: np.ndarray
) -> np.ndarray:
    """inplace calculation of state_in_pauli_basis.
    This corresponds to multiplying the matrix M_nq to (order arranged) dm.
    where M_1q = np.array([[1,  0,  0,  1],  # I dm[ i ][ j ]
                           [0,  1,  1,  0],  # X dm[ i ][j+s]
                           [0, 1j,-1j,  0],  # Y dm[i+s][ j ]
                           [1,  0,  0, -1]]) # Z dm[i+s][j+s]
          M_nq = M_1q \otimes M_1q \otimes ... \otimes M_1q  # n_qubit times


    Args:
        n_qubit (int): the number of qubits.
        density_matrix (np.ndarray): density_matrix of (2 ** n_qubit, 2 ** n_qubit).

    Returns:
        np.ndarray: state_in_pauli_basis of (4 ** n_qubit,).
    """
    # The following code is based on FWHT like method.
    # Reference: https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
    size = 2**n_qubit
    for k in range(n_qubit):
        shift = 1 << k
        for i_offset in range(0, size, shift * 2):
            for j_offset in range(0, size, shift * 2):
                for i in range(i_offset, i_offset + shift):
                    for j in range(j_offset, j_offset + shift):
                        I = density_matrix[i][j] + density_matrix[i + shift][j + shift]
                        Z = density_matrix[i][j] - density_matrix[i + shift][j + shift]
                        density_matrix[i][j] = I
                        density_matrix[i + shift][j + shift] = Z
                        X = density_matrix[i][j + shift] + density_matrix[i + shift][j]
                        Y = (
                            density_matrix[i][j + shift] - density_matrix[i + shift][j]
                        ) * 1j
                        density_matrix[i][j + shift] = X
                        density_matrix[i + shift][j] = Y

    # The following code is based on z order curve.
    # Reference: https://en.wikipedia.org/wiki/Z-order_curve
    interlace_zeros = np.zeros(
        size, dtype=np.int64
    )  # interlace_zeros[0b1011] = 0b1000101
    for i in range(size):
        for s in range(n_qubit):
            if i & (1 << s):
                interlace_zeros[i] |= 1 << (2 * s)
    ret = np.zeros(size * size, np.complex128)
    for i in range(size):
        for j in range(size):
            index = (interlace_zeros[i] << 1) | interlace_zeros[j]
            ret[index] = density_matrix[i][j]
    return ret


def state_in_pauli_basis(state: np.ndarray, check: bool = False):
    basis_count = state.shape[0]
    assert (
        state.shape[0] >= 2 and bin(state.shape[0]).count("1") == 1
    ), "state.shape[0] must be expressed as 2^n, where n is a positive integer."

    n_qubit = state.shape[0].bit_length() - 1
    assert 2**n_qubit == basis_count
    assert state.dtype == np.complex128

    if state.ndim == 1:
        density_matrix = np.outer(state, state.conj())
    elif state.ndim == 2:
        density_matrix = state.copy()
    else:
        raise TypeError()

    ret = _state_in_pauli_basis_inplace_calculation(n_qubit, density_matrix)

    if check:
        n_paulis = _make_n_paulis(n_qubit)
        if state.ndim == 1:
            true_arr = np.array([state.conj() @ pauli @ state for pauli in n_paulis])
        else:
            true_arr = np.array([np.trace(state @ pauli) for pauli in n_paulis])
        assert np.allclose(ret.imag, 0), "ret must be real."
        assert np.allclose(true_arr.imag, 0), "true_arr must be real."
        assert np.allclose(ret.real, true_arr), "ret must be close to true_arr."

    return ret.real
