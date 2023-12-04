from functools import lru_cache
from importlib.resources import files
from typing import Dict, List

import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix

from exputils.stabilizer_group import generator_to_group, pauli_str_to_idx
from exputils.math.fwht import sylvesters


@lru_cache
def get_actual_Amat(sz: int) -> csc_matrix:
    return scipy.sparse.load_npz(
        files("exputils").joinpath(f"../data/Amat/Amat{sz}.npz")
    ).astype(np.int8)


def get_actual_Amats() -> Dict[int, csc_matrix]:
    actual_Amats = dict()
    for sz in [1, 2, 3, 4, 5]:
        actual_Amats[sz] = get_actual_Amat(sz)
    return actual_Amats


def generators_to_Amat(n_qubit: int, generators: List[List[str]]) -> csc_matrix:
    block_sz = 4**n_qubit
    sz = len(generators) * (2**n_qubit)  # width of Amat
    nnz = sz * (2**n_qubit)  # number of non-zero elements of Amat

    rows_per_col = np.zeros(nnz, dtype=np.int32)
    data_per_col = np.zeros(nnz, dtype=np.int8)
    for i, generator in enumerate(generators):
        group = generator_to_group(n_qubit, generator)
        data = []
        rows = []
        for pauli in group:
            data.append(1 if pauli[0] != "-" else -1)
            rows.append(pauli_str_to_idx(n_qubit, pauli.replace("-", "")))
        rows_per_col[i * block_sz : (i + 1) * block_sz] = np.tile(rows, 2**n_qubit)
        data_per_col[i * block_sz : (i + 1) * block_sz] = np.tile(data, 2**n_qubit)

    H = sylvesters(n_qubit)
    indptr = np.arange(sz + 1) * (2**n_qubit)
    indices = rows_per_col
    data = data_per_col * np.tile(H.flatten().astype(np.int8), len(generators))
    return csc_matrix((data, indices, indptr), shape=(4**n_qubit, sz))
