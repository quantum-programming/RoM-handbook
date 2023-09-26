import numpy as np

from exputils.actual_Amat import generators_to_Amat


def check_RoM_result(Amat, rho_vec, coeff, RoM) -> bool:
    assert np.isclose(RoM, np.linalg.norm(coeff, 1))
    assert np.allclose(Amat @ coeff, rho_vec)
    return True


def check_RoM_result_with_cover(
    Amat, rho_vec, coeffs, error_coeffs, error_generators, RoM
) -> bool:
    assert np.isclose(RoM, np.linalg.norm(coeffs, 1) + np.linalg.norm(error_coeffs, 1))
    n_qubit = int(np.log2(len(rho_vec)) / 2)
    assert 4**n_qubit == len(rho_vec)
    error_Amat = generators_to_Amat(n_qubit, error_generators)
    assert np.allclose(Amat @ coeffs + error_Amat @ error_coeffs, rho_vec)
    return True
