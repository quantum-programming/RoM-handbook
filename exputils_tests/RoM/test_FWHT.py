import random
import numpy as np
from exputils.state.random import make_random_quantum_state
from exputils.actual_Amat import get_actual_Amat, generators_to_Amat
from exputils.RoM.fwht import calculate_RoM_FWHT

actual_Amats = {
    n_qubit: get_actual_Amat(n_qubit).T.toarray() for n_qubit in range(1, 4 + 1)
}


def is_in_actual_Amat(n_qubit: int, col: np.ndarray) -> bool:
    """check if the given column is in the actual Amat"""
    for col_of_Amat in actual_Amats[n_qubit]:
        if np.allclose(col, col_of_Amat):
            return True
    assert False, f"{col=} is not in actual_Amat"


def test_FWHT():
    n_qubit = 2
    rho_vec = make_random_quantum_state("pure", n_qubit, 0)
    RoM, coeffs, cover_generators = calculate_RoM_FWHT(n_qubit, rho_vec)
    print(f"{RoM=}", f"{coeffs=}", f"{cover_generators=}", sep="\n")
    Amat = generators_to_Amat(n_qubit, cover_generators)
    print("Amat:", *Amat.toarray().tolist(), sep="\n")
    assert np.allclose(Amat @ coeffs, rho_vec), (Amat @ coeffs, rho_vec)

    for n_qubit in range(1, 4 + 1):
        print(f"{n_qubit=}")
        for seed in range(5 if n_qubit <= 3 else 2):
            rho_vec = make_random_quantum_state("pure", n_qubit, seed)
            RoM, coeffs, cover_generators = calculate_RoM_FWHT(n_qubit, rho_vec)
            Amat = generators_to_Amat(n_qubit, cover_generators)
            assert np.allclose(Amat @ coeffs, rho_vec), (Amat @ coeffs, rho_vec)
            assert all(
                is_in_actual_Amat(n_qubit, Amat.getcol(i).toarray().reshape(-1))
                for i in (
                    range(Amat.shape[1])
                    if n_qubit <= 3
                    else random.sample(range(Amat.shape[1]), 10)
                )
            )
        print(f"{n_qubit=} ok!")

    print("all ok")


if __name__ == "__main__":
    test_FWHT()
