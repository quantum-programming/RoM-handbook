import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix


def khatri_rao_from_Amat(A: csc_matrix, B: csc_matrix) -> csc_matrix:
    """khatri_rao product for Amat

    This function is equivalent to column-wise kronecker product of A and B.
    Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.khatri_rao.html

    Args:
        A (csc_matrix): Amat1
        B (csc_matrix): Amat2

    Returns:
        csc_matrix: khatri_rao product of A and B
    """
    n, k = A.shape
    m, _k = B.shape
    nA = (n.bit_length() - 1) // 2
    nB = (m.bit_length() - 1) // 2
    assert k == _k
    assert A.data.shape == ((2**nA) * k,)
    assert B.data.shape == ((2**nB) * k,)
    A_data = A.data.reshape((k, 2**nA))
    A_idxs = A.indices.reshape((k, 2**nA))
    B_data = B.data.reshape((k, 2**nB))
    B_idxs = B.indices.reshape((k, 2**nB))
    data, rows, cols = khatri_rao_from_data_idxs(A_data, A_idxs, B_data, B_idxs)
    return csc_matrix((data, (rows, cols)), shape=(4 ** (nA + nB), k), dtype=np.int8)


def khatri_rao_from_data_idxs(
    A_data: np.ndarray, A_idxs: np.ndarray, B_data: np.ndarray, B_idxs: np.ndarray
) -> csc_matrix:
    k, n = A_data.shape
    _k, m = B_data.shape
    nA = n.bit_length() - 1
    nB = m.bit_length() - 1
    assert k == _k
    assert A_data.shape == (k, (2**nA))
    assert B_data.shape == (k, (2**nB))
    data = scipy.linalg.khatri_rao(A_data.T, B_data.T).T.flatten()
    cols = np.repeat(np.arange(k), 2 ** (nA + nB))
    rows = np.tile(B_idxs, 2**nA).flatten() + (4**nB) * np.repeat(
        A_idxs.flatten(), 2**nB
    )
    return data, rows, cols
