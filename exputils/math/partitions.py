from functools import lru_cache
from typing import List, Union


@lru_cache()
def partitions(
    n: int, lower_bound: int = 1, upper_bound: Union[int, None] = None
) -> List[List[int]]:
    """It returns a list consisting of lists in ascending order, satisfying the following conditions:
    - The elements of the list are sorted.
    - The elements of the list are between `lower_bound` and `upper_bound`.
    - The sum of the elements of the list is equal to `n`.
    """
    if upper_bound is None:
        upper_bound = n
    assert lower_bound > 0
    if lower_bound > n:
        return []
    ret = []
    for first_element in range(lower_bound, min(n // 2, upper_bound) + 1):
        for remain in partitions(n - first_element, first_element, upper_bound):
            ret.append([first_element] + remain)
    if n <= upper_bound:
        ret.append([n])
    return ret
