from functools import reduce
from typing import List

import numpy as np


def decompose_tensor_product(n_qubit: int, rho_vec: np.ndarray) -> List[np.ndarray]:
    """
    Assume rho_vec = \sigma_1 \otimes \sigma_2 \otimes ... \otimes \sigma_n
    return [sigma_1, sigma_2, ..., sigma_n]
    """
    assert rho_vec.shape == (4**n_qubit,)
    sigmas = [[1, 0, 0, 0] for _ in range(n_qubit)]
    for i in range(n_qubit):
        for j in range(1, 4):
            sigmas[i][j] = rho_vec[(1 << (2 * (n_qubit - 1 - i))) * j]
    ret = list(map(np.array, sigmas))
    assert np.allclose(reduce(np.kron, ret), rho_vec), (reduce(np.kron, ret), rho_vec)
    return ret
