from numba import njit
import numpy as np
from scipy.sparse import csc_matrix
from exputils.perm_Amat import get_row_info


def eff_kron_perm_Amat(n1, perm_Amat_1, n2, perm_Amat_2):
    basis_1 = get_row_info(n1)
    basis_2 = get_row_info(n2)
    basis_result = get_row_info(n1 + n2)
    basis_result_inv = {p: i for i, p in enumerate(basis_result)}

    perm_Amat_1_array = perm_Amat_1.toarray()
    perm_Amat_2_array = perm_Amat_2.toarray()

    perm_Amat_array = np.zeros(
        (len(basis_result), perm_Amat_1.shape[1] * perm_Amat_2.shape[1]), np.int32
    )

    for r1, (x1, y1, z1) in enumerate(basis_1):
        for r2, (x2, y2, z2) in enumerate(basis_2):
            x = x1 + x2
            y = y1 + y2
            z = z1 + z2
            r = basis_result_inv[(x, y, z)]
            perm_Amat_array[r] += np.kron(perm_Amat_1_array[r1], perm_Amat_2_array[r2])

    return csc_matrix(perm_Amat_array)
