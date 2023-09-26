import time
import numpy as np
from functools import reduce
from exputils.state.state_in_pauli_basis import state_in_pauli_basis
from exputils.state.decompose_tensor import decompose_tensor_product
from exputils.state.tensor import make_random_tensor_product_state


def test_decompose_tensor():
    for n_qubit in range(1, 5 + 1):
        for seed in range(5):
            states = []
            for i in range(n_qubit):
                random_complexes = (
                    np.random.RandomState(seed=n_qubit * seed + i).random(2) * 2 - 1
                ) + 1j * (
                    np.random.RandomState(seed=n_qubit * (seed + 1000) + i).random(2)
                    * 2
                    - 1
                )
                random_complexes /= np.linalg.norm(random_complexes, ord=2)
                states.append(random_complexes.copy())

            states_in_pauli_basis = list(map(state_in_pauli_basis, states))
            assert np.allclose(
                reduce(np.kron, states_in_pauli_basis),
                state_in_pauli_basis(reduce(np.kron, states), True),
            )

            rho_vec = reduce(np.kron, states_in_pauli_basis)
            sigmas = decompose_tensor_product(n_qubit, rho_vec)
            assert all(
                np.allclose(sigma, state)
                for sigma, state in zip(sigmas, states_in_pauli_basis)
            )
        print(f"{n_qubit=} ok")
    print("all ok1")

    for n_qubit in range(1, 12 + 1):
        for seed in range(5):
            rho_vec = make_random_tensor_product_state(
                "pure", n_qubit, seed, check=False
            )
            t0 = time.perf_counter()
            sigmas = decompose_tensor_product(n_qubit, rho_vec)
            t1 = time.perf_counter()
            print(f"{n_qubit}: {t1 - t0=}s", end="\r")
            assert np.allclose(reduce(np.kron, sigmas), rho_vec)
        print(f"\n{n_qubit=} ok")
    print("all ok2")


if __name__ == "__main__":
    test_decompose_tensor()
