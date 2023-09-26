from functools import lru_cache
from typing import List

from tqdm.auto import tqdm

from exputils.stabilizer_group import get_valid


def make_next_generators(n: int, generators: List[List[str]]) -> List[List[str]]:
    n += 1
    next_generators = []
    for prev in tqdm(
        generators, total=len(generators), leave=False, desc="make_next_generators"
    ):
        # Enumeration of commutative generators
        pattern1 = [
            list(map(lambda row: row + "I", prev)) + ["I" * (n - 1) + "XYZ"[i]]
            for i in range(3)
        ]
        # Enumeration of anti-commutative generators
        pattern2 = []
        for bit in range(1, 1 << (n - 1)):
            head = [prev[i] + "IY"[(bit >> i) & 1] for i in range(n - 1)]
            antiCommute = get_valid(prev, bit)
            pattern2.append(head + [antiCommute + "X"])
            pattern2.append(head + [antiCommute + "Z"])
        # Add to next_generators
        next_generators.extend(pattern1)
        next_generators.extend(pattern2)
    return sorted(sorted(gen) for gen in next_generators)


@lru_cache
def get_actual_generators(n_qubit: int) -> List[List[str]]:
    if n_qubit == 1:
        return [["X"], ["Y"], ["Z"]]
    else:
        return make_next_generators(n_qubit - 1, get_actual_generators(n_qubit - 1))
