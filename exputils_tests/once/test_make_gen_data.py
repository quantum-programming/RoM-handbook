import time

# import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

from exputils.dot.dot_product import compute_all_dot_products
from exputils.dot.dot_product_with_gen_data import (
    compute_all_dot_products_with_gen_data,
    generate_dot_data_from_gen_data,
)
from exputils.once.make_Amat import is_same_with_actual, make_Amat_from_column_index
from exputils.stabilizer_group import total_stabilizer_group_size
from exputils.state.random import make_random_quantum_state


def test_make_dot_data_7(n_qubit: int, CHUNK_CNT: int):
    if n_qubit <= 4:
        list_of_Amat = []
        for chunk_id in range(CHUNK_CNT):
            data_per_col = []
            rows_per_col = []
            for data, rows in generate_dot_data_from_gen_data(n_qubit, chunk_id):
                data_per_col.append(data)
                rows_per_col.append(rows)
            small_Amat = make_Amat_from_column_index(
                n_qubit,
                np.arange(total_stabilizer_group_size(n_qubit) // CHUNK_CNT),
                np.vstack(data_per_col),
                np.vstack(rows_per_col),
            )
            list_of_Amat.append(small_Amat)
        Amat = scipy.sparse.hstack(list_of_Amat)
        print(f"{Amat.shape=}")
        # if n_qubit <= 3:
        #     plt.imshow(Amat.toarray())
        #     plt.title(f"{n_qubit=}")
        #     plt.show()
        assert is_same_with_actual(n_qubit, Amat)
        print(f"{n_qubit=} passed")
    elif n_qubit <= 6:
        rho_vec = make_random_quantum_state("mixed", n_qubit, seed=0)
        print("computing dots: start")
        t0 = time.perf_counter()
        dots = compute_all_dot_products_with_gen_data(n_qubit, rho_vec, CHUNK_CNT=3)
        t1 = time.perf_counter()
        print(dots[:5])
        print(f"total: {t1-t0=}s")
        dots_true = compute_all_dot_products(n_qubit, rho_vec)
        assert np.allclose(np.sort(dots), np.sort(dots_true))
        print(f"{n_qubit=} passed")
    else:
        assert n_qubit == 7
        pass


if __name__ == "__main__":
    for n_qubit in range(1, 7 + 1):
        test_make_dot_data_7(n_qubit, 3 if n_qubit <= 6 else 225)
