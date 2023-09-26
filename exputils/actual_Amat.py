from functools import lru_cache
from importlib.resources import files
from typing import Dict, List

import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix
from tqdm.auto import tqdm

from exputils.stabilizer_group import generator_to_group, pauli_str_to_idx


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
    data = []
    rows = []
    cols = np.array(
        [np.arange((2**n_qubit) * len(generators)) for _ in range(2**n_qubit)]
    ).T.flatten()
    signs = [
        [
            (1 if bin(sign_bit & idx)[1:].count("1") % 2 == 0 else -1)
            for idx in range(1 << n_qubit)
        ]
        for sign_bit in range(1 << n_qubit)
    ]
    for generator in tqdm(generators, desc="generators_to_Amat", leave=False):
        group = generator_to_group(n_qubit, generator)
        for sign_bit in range(1 << n_qubit):
            for idx, pauli in enumerate(group):
                data.append(signs[sign_bit][idx] * (1 if pauli[0] != "-" else -1))
                rows.append(pauli_str_to_idx(n_qubit, pauli.replace("-", "")))
    return csc_matrix(
        (data, (rows, cols)),
        shape=(1 << (2 * n_qubit), len(generators) * (1 << n_qubit)),
        dtype=np.int8,
    )
