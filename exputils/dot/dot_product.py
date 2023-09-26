import numpy as np
from numba import njit

from exputils.dot.load_data import load_dot_data
from exputils.math.fwht import FWHT


@njit(cache=True)
def _compute_all_dot_products(
    n_qubit: int,
    data_per_col: np.ndarray,
    rows_per_col: np.ndarray,
    rho_vec: np.ndarray,
) -> np.ndarray:
    assert len(data_per_col) == len(rows_per_col)
    arranged_rho_vec = (
        rho_vec[rows_per_col.flatten()] * data_per_col.flatten()
    ).reshape((len(data_per_col), -1))
    ans = np.zeros((len(data_per_col), (1 << n_qubit)), dtype=np.float64)
    for i in range(len(data_per_col)):
        ans[i] = FWHT(n_qubit, arranged_rho_vec[i])
    ans = ans.flatten()
    ans *= 1 << n_qubit
    return ans


def compute_all_dot_products(
    n_qubit: int,
    rho_vec: np.ndarray,
    data_per_col: np.ndarray = None,
    rows_per_col: np.ndarray = None,
) -> np.ndarray:
    """compute the dot product of a rho_vec with all the columns of the actual Amat.

    This function satisfies the following test:

    >>> slow = []
    >>> for col in Amat.T:
    >>>     slow.append(np.dot(rho_vec, col.toarray().flatten()))
    >>> ans = compute_all_dot_products(n_qubit, rho_vec)
    >>> assert np.allclose(ans, slow)

    Args:
        n_qubit (int): the number of qubits
        rho_vec (np.ndarray): rho_vec
        data_per_col (np.ndarray, optional): the data of the actual Amat. Defaults to None.
        rows_per_col (np.ndarray, optional): the rows of the actual Amat. Defaults to None.

    Returns:
        np.ndarray: the dot products.
    """
    if data_per_col is None or rows_per_col is None:
        data_per_col, rows_per_col = load_dot_data(n_qubit)
    return _compute_all_dot_products(n_qubit, data_per_col, rows_per_col, rho_vec)
