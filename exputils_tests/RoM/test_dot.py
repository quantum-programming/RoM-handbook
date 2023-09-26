from exputils.RoM.dot import get_topK_indices
from exputils.once.make_Amat import make_Amat_from_column_index
from exputils.dot.dot_product import compute_all_dot_products
from exputils.dot.load_data import load_dot_data
from exputils.state.random import make_random_quantum_state
from exputils.actual_Amat import get_actual_Amat

import numpy as np


def test_make_Amat_from_column_index():
    for n_qubit in range(1, 4 + 1):
        d, r = load_dot_data(n_qubit)
        K = [1, 1, 1, 0.1][n_qubit - 1]
        for seed in range(3):
            print(f"{(n_qubit,seed)=}")
            rho_vec = make_random_quantum_state("mixed", n_qubit, seed=seed)
            dot_products = compute_all_dot_products(n_qubit, rho_vec, d, r)
            idxs = get_topK_indices(dot_products, K)
            slow = get_actual_Amat(n_qubit)[:, idxs]
            fast = make_Amat_from_column_index(n_qubit, idxs, d, r)
            assert np.allclose(slow.toarray(), fast.toarray())


if __name__ == "__main__":
    test_make_Amat_from_column_index()
