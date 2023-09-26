import random
import numpy as np
from tqdm.auto import tqdm
from time import perf_counter
from qiskit.quantum_info import Pauli
from exputils.stabilizer_group import pauli_str_to_idx
from exputils.dot.pauli_dot import (
    pauli_dot,
    pauli_dot_without_str,
    pauli_dot_without_sign,
)


def test_pauli_dot():
    fast = 0.0
    slow = 0.0
    very_slow = 0.0
    print("Now running...")
    for _ in tqdm(range(1000), leave=False):
        sz = random.randint(1, 20)
        a = ("-" if random.random() < 0.5 else "") + "".join(
            random.choice("IXYZ") for _ in range(sz)
        )
        b = ("-" if random.random() < 0.5 else "") + "".join(
            random.choice("IXYZ") for _ in range(sz)
        )
        t1 = perf_counter()
        phase, s = pauli_dot_without_str(
            (
                2 if a[0] == "-" else 0,
                np.array(["IXYZ".find(c) for c in a.replace("-", "")]),
            ),
            (
                2 if b[0] == "-" else 0,
                np.array(["IXYZ".find(c) for c in b.replace("-", "")]),
            ),
        )
        c1 = ["", "i", "-", "-i"][phase] + "".join(["IXYZ"[c] for c in s])
        t2 = perf_counter()
        phase, s = pauli_dot(
            (2 if a[0] == "-" else 0, a.replace("-", "")),
            (2 if b[0] == "-" else 0, b.replace("-", "")),
        )
        c2 = ["", "i", "-", "-i"][phase] + s
        t3 = perf_counter()
        c3 = str(Pauli.dot(Pauli(a), Pauli(b)))
        t4 = perf_counter()
        assert c1 == c2 == c3, (c1, c2, c3)
        assert pauli_dot_without_sign(
            pauli_str_to_idx(sz, a.replace("-", "")),
            pauli_str_to_idx(sz, b.replace("-", "")),
        ) == pauli_str_to_idx(sz, s)
        fast += t2 - t1
        slow += t3 - t2
        very_slow += t4 - t3

    print("ok!")
    print(f"{fast=}, {slow=}, {very_slow=}")


if __name__ == "__main__":
    test_pauli_dot()
