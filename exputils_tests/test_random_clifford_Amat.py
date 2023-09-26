import qulacs
import numpy as np
from exputils.state.state_in_pauli_basis import state_in_pauli_basis


def generate_random_Amat_Bravyi2021_naive(n_qubit, seeds):
    from random_clifford.group import CliffordCircuitGroupByBravyi2021

    ccg = CliffordCircuitGroupByBravyi2021(n_qubit)
    Amat = []
    for seed in seeds:
        circuit = ccg.get_element(seed)
        state = qulacs.StateVector(n_qubit)  # zero vector
        ccg.simulate_circuit(n_qubit, state, circuit)
        col = state_in_pauli_basis(state.get_vector()).astype(np.float64)
        Amat.append(col)
    return np.array(Amat).T


# since the following code requires clifford sim, we skip this test
# def test_random_clifford_Amat():
#     import random
#     from exputils.random_clifford_Amat import generate_random_Amat_Bravyi2021

#     SEED_MIN = 0
#     SEED_MAX = 2**32 - 1
#     random.seed(13)
#     seed_count = 1000
#     seeds = random.sample(range(SEED_MIN, SEED_MAX), seed_count)
#     for n in range(1, 7):
#         # print(generate_random_Amat_Bravyi2021_naive(n, seeds))
#         # print(generate_random_Amat_Bravyi2021(n, seeds).toarray())
#         assert np.allclose(
#             generate_random_Amat_Bravyi2021_naive(n, seeds),
#             generate_random_Amat_Bravyi2021(n, seeds).toarray(),
#         ), f"\n{generate_random_Amat_Bravyi2021_naive(n, seeds).T}\n\n{generate_random_Amat_Bravyi2021(n, seeds).toarray().astype(np.float64).T}"
#         print(f"{n} ok!")
#     print("all ok!")
