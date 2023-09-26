import numpy as np
from functools import reduce
from exputils.state.canonical_magic_state import (
    make_canonical_magic_state_in_pauli_basis,
)
from exputils.state.random import make_random_quantum_state
from exputils.state.tensor import make_random_tensor_product_state

for n in range(1, 5 + 1):
    print(f"{n=}")
    for seed in range(5):
        print(f"{seed=}")
        H = make_random_tensor_product_state("H", n, seed, check=True)
        print(H)
        assert np.allclose(H, make_canonical_magic_state_in_pauli_basis(n).toarray())
        pure = make_random_tensor_product_state("pure", n, seed, check=True)
        pure2 = reduce(
            np.kron,
            [
                make_random_quantum_state("pure", 1, 1000 * seed + i, check=True)
                for i in range(n)
            ],
        )
        print(pure)
        assert np.allclose(pure, pure2)
        mixed = make_random_tensor_product_state("mixed", n, seed, check=True)
        mixed2 = reduce(
            np.kron,
            [
                make_random_quantum_state("mixed", 1, 1000 * seed + i, check=True)
                for i in range(n)
            ],
        )
        print(mixed)
        assert np.allclose(mixed, mixed2)

for n in range(6, 12 + 1):
    print(f"{n=}")
    for seed in range(5):
        print(make_random_tensor_product_state("pure", n, seed, check=False))
        print(make_random_tensor_product_state("mixed", n, seed, check=False))
