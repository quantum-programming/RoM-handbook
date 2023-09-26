import numpy as np
from exputils.dot.dot_product import compute_all_dot_products


def lb_by_dot(n_qubit: int, rho_vec: np.ndarray) -> float:
    """lower bound of RoM calculated by dot product

    Args:
        n_qubit (int): number of qubits
        rho_vec (np.ndarray): state

    Returns:
        float: lower bound of RoM
    """
    dots = compute_all_dot_products(n_qubit, rho_vec)
    dp = np.max(dots)
    dm = np.min(dots)
    return 1 + 2 * (np.linalg.norm(rho_vec, ord=2) ** 2 - dp) / (dp - dm)


def main():
    from exputils.state.canonical_magic_state import (
        make_canonical_magic_state_in_pauli_basis,
    )
    from exputils.st_norm import lb_by_st_norm

    n_qubit = 6
    rho_vec = make_canonical_magic_state_in_pauli_basis(n_qubit).toarray().flatten()
    print(f"{lb_by_st_norm(n_qubit, rho_vec)=}")
    print(f"{lb_by_dot(n_qubit, rho_vec)=}")


if __name__ == "__main__":
    main()
