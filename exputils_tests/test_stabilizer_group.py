import time
import random
from typing import List, Tuple
from qiskit.quantum_info import Pauli
from exputils.dot.pauli_dot import pauli_dot
from exputils.actual_generators import get_actual_generators
from exputils.stabilizer_group import (
    generator_to_group,
    generator_to_group_in_phase_pidx,
    idx_to_pauli_str,
)


def _check_generator_to_group(n_qubit: int, generator: List[str]) -> Tuple[float]:
    def pauliProd(n_qubit, paulis):
        p = Pauli("I" * n_qubit)
        for pauli in paulis:
            p = Pauli.dot(p, pauli)
        return p

    def slow_generator_to_group(n_qubit: int, generator: List[str]) -> List[str]:
        gens = [
            ((2, gen_str[1:]) if gen_str[0] == "-" else (0, gen_str))
            for gen_str in generator
        ]
        group = [(0, "I" * n_qubit)]
        for gen in gens[::-1]:
            group = sum(([g, pauli_dot(g, gen)] for g in group), [])
        assert all(phase == 0 or phase == 2 for phase, _ in group), group
        group = [("" if phase == 0 else "-") + s for phase, s in group]

        return group

    def very_slow_generator_to_group(n_qubit: int, generator: List[str]):
        # generators2Amatで符号計算する関係上、以下のslowGroupに示すような順序である必要がある

        gens = [Pauli(gen) for gen in generator]
        slowGroup = [
            str(pauliProd(n_qubit, [gens[i] for i in range(n_qubit) if bit & (1 << i)]))
            for bit in range(1 << n_qubit)
        ]
        return slowGroup

    t0 = time.perf_counter()
    g_phase, g_pidx = generator_to_group_in_phase_pidx(n_qubit, generator)
    very_fast_group = [
        ("-" if p == 2 else "") + idx_to_pauli_str(n_qubit, pidx)
        for p, pidx in zip(g_phase, g_pidx)
    ]
    t1 = time.perf_counter()
    fast_group = generator_to_group(n_qubit, generator)
    t2 = time.perf_counter()
    slow_group = slow_generator_to_group(n_qubit, generator)
    t3 = time.perf_counter()
    very_slow_group = very_slow_generator_to_group(n_qubit, generator)
    t4 = time.perf_counter()
    assert len(set(very_fast_group)) == 2 ** len(generator)
    assert very_fast_group == fast_group == slow_group == very_slow_group
    return t1 - t0, t2 - t1, t3 - t2, t4 - t3


def test_stabilizer_group():
    for n_qubit in range(1, 4 + 1):
        print(f"{n_qubit=}")
        gens = get_actual_generators(n_qubit)
        very_fast_time = 0
        fast_time = 0
        slow_time = 0
        very_slow_time = 0
        for gen in gens[: min(100, len(gens))]:
            ff, f, s, ss = _check_generator_to_group(
                n_qubit, [("-" if random.random() < 0.5 else "") + g for g in gen]
            )
            very_fast_time += ff
            fast_time += f
            slow_time += s
            very_slow_time += ss
        print(f"{very_fast_time=}, {fast_time=}, {slow_time=}, {very_slow_time=}")


if __name__ == "__main__":
    test_stabilizer_group()
