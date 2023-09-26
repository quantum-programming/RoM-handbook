import numpy as np
from numba import njit


# https://ja.wikipedia.org/wiki/%E3%82%A2%E3%83%80%E3%83%9E%E3%83%BC%E3%83%AB%E5%A4%89%E6%8F%9B
def sylvesters(n_qubit: int) -> np.ndarray:
    """Make a Hadamard matrix. Note that the normalization coefficient is omitted."""
    assert n_qubit >= 0
    if n_qubit == 0:
        return np.array([1])
    if n_qubit == 1:
        return np.array([[1, 1], [1, -1]])
    return np.kron(np.array([[1, 1], [1, -1]]), sylvesters(n_qubit - 1))


# https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
@njit(cache=True)
def FWHT(N: int, As: np.ndarray):
    """Compute H @ As / (1<<N) with Fast Walsh Hadamard Transform"""
    assert As.shape[0] == 1 << N
    inplaceAs = As.copy()
    h = 1
    while h < len(inplaceAs):
        for i in range(0, len(inplaceAs), h << 1):
            for j in range(i, i + h):
                x = inplaceAs[j]
                y = inplaceAs[j + h]
                inplaceAs[j] = x + y
                inplaceAs[j + h] = x - y
        h <<= 1
    return inplaceAs / (1 << N)
