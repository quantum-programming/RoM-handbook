from typing import List

import numpy as np
from numba import njit


# Reference: https://drken1215.hatenablog.com/entry/2019/03/20/202800
@njit(cache=True)
def gauss_jordan_get_only_rank(A: np.ndarray, m: int) -> int:
    """By using Gauss-Jordan elimination, get the rank of A"""
    rank = 0
    for col in range(m):
        pivot = -1
        for row in range(rank, A.shape[0]):
            if (A[row] >> col) & 1:
                pivot = row
                break
        if pivot == -1:
            continue
        A[pivot], A[rank] = A[rank], A[pivot]
        for row in range(A.shape[0]):
            if row != rank and (A[row] >> col) & 1:
                A[row] ^= A[rank]
        rank += 1
    return rank


# Reference: https://drken1215.hatenablog.com/entry/2019/03/20/202800
@njit(cache=True)
def gauss_jordan(A: np.ndarray, m: int) -> List[int]:
    """Solve Ax=b by using Gauss-Jordan elimination

    Args:
        A (np.ndarray): the representation of (A|b) in bit
        m (int): the number of A's columns

    Returns:
        List[int]: the list of x
    """
    rank = 0
    for col in range(m):
        pivot = -1
        for row in range(rank, A.shape[0]):
            if (A[row] >> col) & 1:
                pivot = row
                break
        if pivot == -1:
            continue
        A[pivot], A[rank] = A[rank], A[pivot]
        for row in range(A.shape[0]):
            if row != rank and (A[row] >> col) & 1:
                A[row] ^= A[rank]
        rank += 1
    for row in range(rank, A.shape[0]):
        assert A[row] & (1 << m) == 0
    ret = [0 for _ in range(m)]
    col = 0
    for row in range(rank):
        while (A[row] >> col) & 1 == 0:
            col += 1
        ret[col] = (A[row] >> m) & 1
    return ret


def solve_f2_linear_equation(A: List[List[int]], b: int) -> List[int]:
    """Solve Ax=b and get one of the solutions arbitrarily

    Args:
        A (List[List[int]]): the A of Ax=b
        b (int): the b of Ax=b

    Returns:
        List[int]: the x of Ax=b
    """
    n, m = len(A), len(A[0])
    assert 0 <= b < (1 << n)
    M = np.array(
        [
            sum((A[i][j] if j != m else ((b >> i) & 1)) << j for j in range(m + 1))
            for i in range(n)
        ]
    )
    return gauss_jordan(M, m)


def main():
    A = [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0]]
    b = 0b001  # [1, 0, 0]
    x = solve_f2_linear_equation(A, b)
    assert gauss_jordan_get_only_rank(
        np.array([sum(val << idx for idx, val in enumerate(row)) for row in A]), 4
    ) == 3 and x == [1, 1, 0, 0]

    for _ in range(100):
        n = np.random.randint(1, 10 + 1)
        m = n + np.random.randint(1, 10 + 1)
        A = np.random.randint(2, size=(n, m))
        b = np.random.randint(1 << n)
        A_in_bit = np.array([sum(A[i][j] << j for j in range(m)) for i in range(n)])

        print(gauss_jordan_get_only_rank(A_in_bit, m))
        if gauss_jordan_get_only_rank(A_in_bit, m) != n:
            print(f"failed, {A=}")
        else:
            x = np.array(solve_f2_linear_equation(A, b))
            print(f"{A=}, {b=}, {x=}")
            assert all(np.dot(A[i], x) % 2 == ((b >> i) & 1) for i in range(n))


if __name__ == "__main__":
    main()
