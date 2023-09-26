import math

import numpy as np

from exputils.actual_Amat import get_actual_Amat
from exputils.divide import (
    divide_to_parts,
    divide_generator,
    new_idxs_by_division,
    rev_new_idxs_by_division,
)


def test_divide():
    assert divide_to_parts(3, (1, 2)) == [
        ((0,), (1, 2)),
        ((0, 2), (1,)),
        ((0, 1), (2,)),
    ]
    assert divide_to_parts(4, (2, 2)) == [
        ((0, 2), (1, 3)),
        ((0, 1), (2, 3)),
        ((0, 3), (1, 2)),
    ]
    for k in range(1, 3 + 1):
        for n in range(k, 3 * k + 1, k):
            assert len(divide_to_parts(n, (k,) * (n // k))) == math.factorial(n) // (
                math.factorial(k) ** (n // k)
            ) // math.factorial(n // k)


def test_divide_generator():
    n = 3
    div_sizes = (1, 2)
    actual = divide_to_parts(n, div_sizes)
    for div in divide_generator(n, div_sizes, 10):
        assert len(div) == 2
        assert div in actual

    n = 21
    div_sizes = (3,) * 7
    cnt = 10
    for div in divide_generator(n, div_sizes, cnt):
        assert len(div) == 7


def test_new_idxs_by_division():
    division = ((1,), (0,))
    assert new_idxs_by_division(division).tolist() == [
        0,
        4,
        8,
        12,
        1,
        5,
        9,
        13,
        2,
        6,
        10,
        14,
        3,
        7,
        11,
        15,
    ]

    Amat = get_actual_Amat(2)
    new_idxs = new_idxs_by_division(((1, 0),))
    rev_idxs = rev_new_idxs_by_division(new_idxs)
    assert np.allclose(Amat.toarray(), (Amat[new_idxs])[rev_idxs].toarray())

    Amat = get_actual_Amat(3)
    new_idxs = new_idxs_by_division(((1, 2), (0,)))
    rev_idxs = rev_new_idxs_by_division(new_idxs)
    assert np.allclose(Amat.toarray(), (Amat[new_idxs])[rev_idxs].toarray())


if __name__ == "__main__":
    test_divide()
    test_divide_generator()
    test_new_idxs_by_division()
