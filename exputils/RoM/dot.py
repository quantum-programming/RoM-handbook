from typing import Tuple, Union

import numpy as np
from scipy.sparse import csc_matrix

from exputils.once.make_Amat import make_Amat_from_column_index
from exputils.dot.dot_product import compute_all_dot_products
from exputils.dot.load_data import load_dot_data
from exputils.RoM.custom import calculate_RoM_custom


def get_topK_indices(vals: np.ndarray, K: float) -> np.ndarray:
    """get indices of top (100*K/2)% and bottom (100*K/2)% elements of vals.

    Args:
        vals (np.ndarray): the array to be selected from.
        K (float): the ratio of top and bottom elements to be selected.

    Returns:
        np.ndarray: indices of selected elements.
    """
    assert 0 < K <= 1
    if K == 1:
        return np.arange(len(vals))
    num = int(len(vals) * K)
    indexes1 = np.argpartition(-vals, num // 2)[: num // 2]
    indexes2 = np.argpartition(vals, num - num // 2)[: num - num // 2]
    return np.concatenate((indexes1, indexes2))


def get_topK_Amat(
    n_qubit: int,
    rho_vec: np.ndarray,
    K: float,
    data_per_col: np.ndarray = None,
    rows_per_col: np.ndarray = None,
) -> csc_matrix:
    """get Amat of top (100*K/2)% and bottom (100*K/2)% col with respect to dot_products.

    Args:
        n_qubit (int): the number of qubits. (1 <= n_qubit <= 7)
        rho_vec (np.ndarray): the rho_vec.
        K (float): the ratio of top and bottom elements to be selected.

    Returns:
        csc_matrix: the selected Amat.
    """
    assert 0 < K <= 1
    assert 1 <= n_qubit <= 7

    if n_qubit <= 6:
        if data_per_col is None or rows_per_col is None:
            data_per_col, rows_per_col = load_dot_data(n_qubit)

        # the following two lines are bottleneck even if n_qubit == 6
        dot_products = compute_all_dot_products(
            n_qubit, rho_vec, data_per_col, rows_per_col
        )
        idxs = get_topK_indices(dot_products, K)

        return make_Amat_from_column_index(n_qubit, idxs, data_per_col, rows_per_col)
    else:
        raise NotImplementedError
        assert os.path.exists(
            f"../../data/genData/{n_qubit}_0.npz"
        ), "if n_qubit == 7, you should generate the genData first."

        get_topK_Amat_from_gen_data(n_qubit, rho_vec, K)


def calculate_RoM_dot(
    n_qubit: int,
    rho_vec: np.ndarray,
    K: float = None,
    verbose: bool = False,
    method: str = "scipy",
    return_dual: bool = False,
    crossover: bool = True,
    presolve: bool = False,
) -> Union[
    Tuple[float, np.ndarray, csc_matrix],
    Tuple[float, np.ndarray, csc_matrix, np.ndarray],
]:
    """calculate RoM with dot product method"""
    assert 1 <= n_qubit <= 6
    if K is None:
        K = [1.0, 1.0, 1.0, 0.1, 0.01, 0.0001][n_qubit - 1]
    Amat = get_topK_Amat(n_qubit, rho_vec, K)
    RoM, *other = calculate_RoM_custom(
        Amat, rho_vec, verbose, method, return_dual, crossover, presolve
    )
    if not np.isnan(RoM):
        if return_dual:
            assert len(other) == 2
            coeff, dual = other
        else:
            assert len(other) == 1
            coeff = other[0]
        valid_indices = np.abs(coeff) > 1e-10
        if return_dual:
            return RoM, coeff[valid_indices], Amat[:, valid_indices], dual
        else:
            return RoM, coeff[valid_indices], Amat[:, valid_indices]
    else:
        if return_dual:
            return np.nan, np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan, np.nan
