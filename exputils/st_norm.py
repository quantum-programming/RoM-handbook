import math
from typing import Union

import numpy as np
from scipy.sparse import coo_matrix


def compute_st_norm(rho_vec: Union[np.ndarray, coo_matrix]) -> float:
    """st-norm

    Args:
        rho_vec (Union[np.ndarray, coo_matrix]): state in Pauli basis

    Returns:
        float: st_norm
    """
    if isinstance(rho_vec, np.ndarray):
        assert len(rho_vec.shape) == 1
        n = math.log(rho_vec.shape[0], 4)
        assert 4**n == rho_vec.shape[0]
        return np.sum(np.abs(rho_vec)) / (2**n)
    elif isinstance(rho_vec, coo_matrix):
        assert len(rho_vec.shape) == 2 and rho_vec.shape[0] == 1
        n = math.log(rho_vec.shape[1], 4)
        assert 4**n == rho_vec.shape[1]
        return coo_matrix.sum(abs(rho_vec)) / (2**n)
    else:
        raise TypeError()


def lb_by_st_norm(n_qubit: int, rho_vec: Union[np.ndarray, coo_matrix]) -> float:
    """calculate the lower bound of RoM by st_norm

    If $ρ$ is an n-qubit state then
    \frac{\mathcal{D}(ρ) - \frac{1}{2^n}}{\left(1 - \frac{1}{2^n}\right)} \leq \mathcal{R}(ρ)


    Args:
        n_qubit (int): the number of qubits
        rho_vec (Union[np.ndarray, coo_matrix]): state in Pauli basis

    Returns:
        float: the lower bound of RoM by st_norm
    """
    return (compute_st_norm(rho_vec) - 1 / (2**n_qubit)) / (1 - 1 / (2**n_qubit))


def main():
    from exputils.state.canonical_magic_state import (
        make_canonical_magic_state_in_pauli_basis,
    )

    accurate_Hs = [
        math.sqrt(2),
        (1 + 3 * math.sqrt(2)) / 3,
        (1 + 4 * math.sqrt(2)) / 3,
        (3 + 8 * math.sqrt(2)) / 5,
        3.68705,
    ]

    for n_qubit in range(1, 5 + 1):
        print("=" * 10)
        print(f"{n_qubit=}")
        rho_vec = (
            make_canonical_magic_state_in_pauli_basis(n_qubit).toarray().reshape(-1)
        )
        print(f"{compute_st_norm(rho_vec)=}")
        print(f"{lb_by_st_norm(n_qubit, rho_vec)=}")
        print(f"{accurate_Hs[n_qubit - 1]=}")
        assert lb_by_st_norm(n_qubit, rho_vec) < accurate_Hs[n_qubit - 1]


if __name__ == "__main__":
    main()
