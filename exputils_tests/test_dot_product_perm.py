import os
import warnings
from importlib.resources import files

import numpy as np

from exputils.perm_Amat import get_perm_Amat, get_row_size
from exputils.perm_dot import compute_all_dot_products_perm


def test_dot_product():
    rng = np.random.default_rng(0)
    for n in range(1, 7 + 1):
        if not os.path.exists(
            files("exputils").joinpath(f"../data/LCgraphs/perm_Amat_{n}.npz")
        ):
            warnings.warn(
                f"LCgraphs/perm_Amat_{n}.npz does not exist."
                "If you need the data, please run make_perm_Amat_by_LCgraphs.py"
                " in the root directory of stabilizer simulator."
            )
            continue
        perm_Amat = get_perm_Amat(n)
        row_size = get_row_size(n)
        test_count = 10 ** ((7 - n) // 2)
        print(f"{n = }, {test_count = }")
        for _ in range(test_count):
            vec = rng.normal(size=row_size)
            dots_expected = vec @ perm_Amat
            dots_actual = compute_all_dot_products_perm(n, vec)
            # print(dots_expected)
            # print(dots_actual)
            assert np.allclose(dots_expected, dots_actual)
        print(f"{n = } ok!")


if __name__ == "__main__":
    test_dot_product()
