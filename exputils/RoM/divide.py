import time
import warnings
from functools import reduce
from typing import List, Tuple

import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix

from exputils.RoM.dot import calculate_RoM_dot
from exputils.state.decompose_tensor import decompose_tensor_product
from exputils.divide import divide_to_parts, new_idxs_by_division


def _solve_small_element(
    states: List[np.ndarray], elem: Tuple[int], K: float, verbose: bool, method: str
):
    rho_vec = np.copy(reduce(np.kron, [states[i] for i in elem]))
    RoM, coeff, Amat = calculate_RoM_dot(
        len(elem), rho_vec, K=K, verbose=verbose, method=method
    )
    if np.isnan(RoM):
        return np.inf, np.array([0]), csc_matrix([[0]])
    else:
        Amat = Amat[:, np.abs(coeff) > 1e-10]
        coeff = coeff[np.abs(coeff) > 1e-10]
        assert np.allclose(Amat @ coeff, rho_vec)
        return RoM, coeff, Amat


def calculate_RoM_divide(
    n_qubit: int,
    div_sizes: Tuple[int],
    rho_vec: np.ndarray,
    method: str,
    TL: float = 60.0,
):
    states = decompose_tensor_product(n_qubit, rho_vec)
    assert np.allclose(rho_vec, reduce(np.kron, states))

    divs = divide_to_parts(n_qubit, div_sizes)
    np.random.shuffle(divs)

    best_RoM = np.inf
    best_div_idx = -1
    div_to_elem_answer = dict()

    time_start = time.perf_counter()
    for div_idx, div in enumerate(divs):
        if time.perf_counter() - time_start > TL:
            warnings.warn("Time Limit Exceeded (calculate RoM divide)")
            break
        RoM = 1.0
        for elem in div:
            if elem not in div_to_elem_answer:
                div_to_elem_answer[elem] = _solve_small_element(
                    states, elem, K=None, verbose=False, method=method
                )
            RoM_elem = div_to_elem_answer[elem][0]
            RoM *= RoM_elem
            if RoM > best_RoM:
                break
        if RoM < best_RoM:
            best_RoM = RoM
            best_div_idx = div_idx

    if best_div_idx == -1:
        raise ValueError("No solution found")

    best_div = divs[best_div_idx]
    best_Amat = csc_matrix([[1]])
    best_coeff = np.array([1.0])
    for elem in best_div:
        elem_coeff = div_to_elem_answer[elem][1]
        elem_Amat = div_to_elem_answer[elem][2]
        best_coeff = np.kron(best_coeff, elem_coeff)
        best_Amat = scipy.sparse.kron(best_Amat, elem_Amat, format="csc")
    new_idxs = new_idxs_by_division(best_div)
    best_Amat = best_Amat[new_idxs]
    best_Amat = best_Amat[:, np.abs(best_coeff) > 1e-10]
    best_coeff = best_coeff[np.abs(best_coeff) > 1e-10]
    assert np.isclose(np.sum(np.abs(best_coeff)), best_RoM)
    assert np.allclose(best_Amat @ best_coeff, rho_vec, atol=1e-5)

    return best_RoM, best_coeff, best_Amat
