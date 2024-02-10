from itertools import combinations
from typing import Generator, List, Set, Tuple

import numpy as np
import random
from numba import njit


def _dfs_for_divide(
    left: Set[int],
    now: List[Tuple[int]],
    div_idx: int,
    div_sizes: Tuple[int],
    ans: Set[Tuple[int]],
    same: bool,
):
    if len(left) == div_sizes[div_idx]:
        ans.add(tuple(sorted(now + [tuple(left)])))
        return
    if same:
        first = left.pop()
        for _new_elem in combinations(left, div_sizes[div_idx] - 1):
            new_elem = [(first,) + _new_elem]
            _dfs_for_divide(
                left - set(_new_elem), now + new_elem, div_idx + 1, div_sizes, ans, same
            )
    else:
        for new_elem in combinations(left, div_sizes[div_idx]):
            _dfs_for_divide(
                left - set(new_elem),
                now + [tuple(new_elem)],
                div_idx + 1,
                div_sizes,
                ans,
                same,
            )


def divide_to_parts(n: int, div_sizes: Tuple[int]) -> List[Tuple[Tuple[int]]]:
    """divide n into several parts.

    Args:
        n (int): the number to be divided.
        div_sizes (Tuple[int]): the sizes of the parts.

    Returns:
        List[Tuple[Tuple[int]]]: the division.

    Examples:
        >>> divide(3, (1, 2))
        [((0,), (1, 2)), ((0, 2), (1,)), ((0, 1), (2,))]
        >>> divide(4, (2, 2))
        [((0, 2), (1, 3)), ((0, 1), (2, 3)), ((0, 3), (1, 2))]

    """
    assert sum(div_sizes) == n and all(d > 0 for d in div_sizes)
    ans = set()
    _dfs_for_divide(set(range(n)), [], 0, div_sizes, ans, len(set(div_sizes)) == 1)
    return list(ans)


def divide_generator(
    n: int, div_sizes: Tuple[int], cnt: int
) -> Generator[Tuple[Tuple[int]], None, None]:
    assert sum(div_sizes) == n and all(d > 0 for d in div_sizes)

    for _ in range(cnt):
        left = set(range(n))
        now = []
        for sz in div_sizes:
            new_elem = tuple(sorted(random.sample(list(left), sz)))
            left -= set(new_elem)
            now.append(new_elem)
        yield tuple(sorted(now))


def new_idxs_by_division(div: Tuple[Tuple[int]]) -> np.ndarray:
    """calculate the idxs for reordering states.

    Args:
        div (Tuple[Tuple[int]]): the division.

    Returns:
        np.ndarray: the idxs for reordering states.

    Examples:
        >>> division = ((1,), (0,))
        >>> new_idxs_by_division(division)
        [ 0  4  8 12  1  5  9 13  2  6 10 14  3  7 11 15]
    """
    order = sum(div, ())
    n_qubit = len(order)
    assert max(order) == n_qubit - 1 and min(order) == 0
    ret = _new_idxs_by_division(order, n_qubit)
    assert np.unique(ret).size == ret.size
    return ret


@njit(cache=True)
def _new_idxs_by_division(order: Tuple[int], n_qubit: int) -> np.ndarray:
    new_idxs = np.zeros(4**n_qubit, dtype=np.int64)
    for _i in range(4**n_qubit):
        i = _i
        new_i = 0
        for j in range(n_qubit):
            new_i += (i & 0b11) << (2 * (n_qubit - 1 - order[n_qubit - 1 - j]))
            i >>= 2
        new_idxs[new_i] = _i
    return new_idxs


def rev_new_idxs_by_division(new_idxs: np.ndarray) -> np.ndarray:
    """calculate the reverse idxs for reordering states.

    Args:
        new_idxs (np.ndarray): the idxs for reordering states.

    Returns:
        np.ndarray: the reverse idxs for reordering states.
    """
    rev_idxs = np.zeros_like(new_idxs)
    rev_idxs[new_idxs] = np.arange(len(new_idxs))
    return rev_idxs


if __name__ == "__main__":
    print(divide_to_parts(3, (1, 2)))
    print(divide_to_parts(4, (2, 2)))
    print(new_idxs_by_division(((1,), (0,))))
    print(rev_new_idxs_by_division(np.array([1, 2, 0])))
