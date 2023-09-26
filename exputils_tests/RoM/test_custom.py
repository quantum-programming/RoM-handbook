import numpy as np
from time import perf_counter
from scipy.sparse import csr_matrix
from exputils.actual_Amat import get_actual_Amat
from exputils.RoM.custom import calculate_RoM_custom
from exputils.state.canonical_magic_state import (
    make_canonical_magic_state_in_pauli_basis,
)


def checkSpeed():
    Amat = get_actual_Amat(4)
    rho_vec = make_canonical_magic_state_in_pauli_basis(4).toarray()

    for method in ["scipy", "gurobi"]:
        t0 = perf_counter()
        RoM, _ = calculate_RoM_custom(Amat, rho_vec, method=method)
        t1 = perf_counter()
        print(f"{method:<6}: RoM={RoM} time={t1 - t0}")


def checkAccuracy():
    stab_states = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, -1, 0, 0, 0, 0],
            [0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 1, -1],
        ]
    )
    rho_vec = np.array([1, 0, 0, 0])
    RoM1, coeff1 = calculate_RoM_custom(
        csr_matrix(stab_states), rho_vec, method="gurobi"
    )
    RoM2, coeff2 = calculate_RoM_custom(stab_states, rho_vec, method="scipy")
    assert np.isclose(RoM1, 1)
    assert np.isclose(RoM2, 1)
    assert np.allclose(stab_states @ coeff1, rho_vec), stab_states @ coeff1
    assert np.allclose(stab_states @ coeff2, rho_vec), stab_states @ coeff2
    print(f"{RoM1=}, {coeff1=}")

    rho_vec = np.array([0, 2, -1, 0])
    RoM1, coeff1 = calculate_RoM_custom(
        csr_matrix(stab_states), rho_vec, method="gurobi"
    )
    RoM2, coeff2 = calculate_RoM_custom(stab_states, rho_vec, method="scipy")
    assert np.isclose(RoM1, 3)
    assert np.isclose(RoM2, 3)
    assert np.allclose(stab_states @ coeff1, rho_vec), stab_states @ coeff1
    assert np.allclose(stab_states @ coeff2, rho_vec), stab_states @ coeff2
    print(f"{RoM1=}, {coeff1=}")


def test_custom():
    checkAccuracy()
    checkSpeed()
    print("ok!")


if __name__ == "__main__":
    test_custom()
