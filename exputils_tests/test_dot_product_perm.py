import time
import numpy as np
from tqdm.auto import tqdm
from exputils.perm_Amat import get_perm_Amat, get_row_size
from exputils.perm_dot import compute_all_dot_products_perm
from exputils.once.make_perm_Amat_by_LCgraphs import (
    make_perm_Amat_from_group_to_perm,
)

from exputils.math.fwht import sylvesters


def test_dot_product():
    rng = np.random.default_rng(0)
    for n in range(1, 7 + 1):
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
