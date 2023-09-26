import lzma
import os
from typing import List

import numpy as np
from tqdm.auto import tqdm

from exputils.actual_generators import get_actual_generators
from exputils.stabilizer_group import generator_to_group, pauli_str_to_idx


def generators_to_info(n_qubit: int, generators: List[List[str]]):
    assert np.iinfo(np.uint16).min <= 0 and 4**n_qubit <= np.iinfo(np.uint16).max
    data_per_col = np.zeros((len(generators), 2**n_qubit), dtype=np.int8)
    rows_per_col = np.zeros((len(generators), 2**n_qubit), dtype=np.uint16)
    for i, generator in tqdm(
        enumerate(generators),
        total=len(generators),
        desc="generators_to_info",
        leave=False,
    ):
        group = generator_to_group(n_qubit, generator)
        data_per_col[i] = np.array([1 if pauli[0] != "-" else -1 for pauli in group])
        rows_per_col[i] = np.array(
            [pauli_str_to_idx(n_qubit, pauli.replace("-", "")) for pauli in group]
        )
    return data_per_col, rows_per_col


def main():
    assert os.path.exists("data/dotData")

    for n_qubit in range(1, 6 + 1):
        print(f"{n_qubit=}")
        generators = get_actual_generators(n_qubit)
        data_per_col, rows_per_col = generators_to_info(n_qubit, generators)
        path = f"data/dotData/dotData{n_qubit}.npz"
        np.savez_compressed(path, data_per_col=data_per_col, rows_per_col=rows_per_col)
        CHUNK_SIZE = 10 * 1024 * 1024  # 1MB
        with open(path, "rb") as input:
            with lzma.open(path + ".xz", "w") as output:
                while chunk := input.read(CHUNK_SIZE):
                    output.write(chunk)
                    output.flush()


if __name__ == "__main__":
    main()
