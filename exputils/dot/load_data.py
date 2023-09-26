import os
from importlib.resources import files
from typing import Tuple

import numpy as np


def load_dot_data(n_qubit: int) -> Tuple[np.ndarray, np.ndarray]:
    path = files("exputils").joinpath(f"../data/dotData/")

    if not os.path.exists(path.joinpath(f"dotData{n_qubit}.npz")):
        import lzma

        assert os.path.exists(path.joinpath(f"dotData{n_qubit}.npz.xz"))
        with lzma.open(path.joinpath(f"dotData{n_qubit}.npz.xz"), "rb") as f:
            np.savez_compressed(path.joinpath(f"dotData{n_qubit}.npz"), **np.load(f))

    loaded = np.load(path.joinpath(f"dotData{n_qubit}.npz"))
    data_per_col = loaded["data_per_col"]
    rows_per_col = loaded["rows_per_col"]
    del loaded

    return data_per_col, rows_per_col
