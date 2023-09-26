import os
from collections import Counter
from functools import reduce

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from exputils.actual_Amat import get_actual_Amat
from exputils.actual_generators import get_actual_generators
from exputils.dot.dot_product import compute_all_dot_products
from exputils.dot.load_data import load_dot_data
from exputils.once.make_dot_data import generators_to_info
from exputils.stabilizer_group import total_stabilizer_group_size


def format_int(x) -> str:
    if x == 0:
        return ""
    if isinstance(x, float):
        return f"{x:.3f}"
    elif x < 1e9:
        return str(x)
    else:
        return f"{x:.2e}"


# def fast_compute_abs_dot(n_qubit, col):
#     _, rows_per_col = load_dot_data(n_qubit)
#     abs_Amat = csc_matrix(
#         (
#             np.ones(rows_per_col.size),
#             (
#                 rows_per_col.flatten(),
#                 np.array(
#                     [np.arange(len(rows_per_col)) for _ in range(2**n_qubit)]
#                 ).T.flatten(),
#             ),
#         )
#     )
#     dots_abs = np.abs(col).T @ abs_Amat
#     return Counter(np.array([dots_abs for _ in range(2**n_qubit)]).flatten())


def check():
    print("now checking...")
    for n_qubit in range(1, 4 + 1):
        print(n_qubit)
        Amat = get_actual_Amat(n_qubit)
        res1 = None
        res2 = None
        for cnt, col in enumerate(Amat.toarray().T):
            dots = compute_all_dot_products(n_qubit, col)
            assert np.allclose(dots, col.T @ Amat)
            # dots_abs = np.abs(col).T @ np.abs(Amat.toarray())
            if res1 is None or res2 is None:
                res1 = Counter(dots)
                # res2 = Counter(dots_abs)
                # assert res2 == fast_compute_abs_dot(n_qubit, col)
            else:
                assert res1 == Counter(dots)
                # assert res2 == Counter(dots_abs)
                # assert res2 == fast_compute_abs_dot(n_qubit, col)
            if n_qubit >= 4 and cnt >= 3:
                break
    print("done")


# check()

maxN = 6

df1 = pd.DataFrame(
    columns=range(1, maxN + 1),
    index=["n qubit", "0", "1", "2", "4", "8", "16", "32", "64"],
)

for n_qubit in range(1, maxN + 1):
    print(n_qubit)
    rho_vec = reduce(np.kron, [np.array([1, 1, 0, 0]) for _ in range(n_qubit)])

    if n_qubit <= 6:
        data_per_col, rows_per_col = load_dot_data(n_qubit)
    else:
        generators = get_actual_generators(n_qubit)
        data_per_col, rows_per_col = generators_to_info(n_qubit, generators)

    dots = compute_all_dot_products(n_qubit, rho_vec, data_per_col, rows_per_col)
    res1 = Counter(dots)

    if n_qubit <= 5:
        assert np.allclose(dots, rho_vec.T @ get_actual_Amat(n_qubit))
    if n_qubit <= 4:
        dots_abs = np.abs(rho_vec).T @ np.abs(get_actual_Amat(n_qubit).toarray())
    #     assert Counter(dots_abs) == fast_compute_abs_dot(n_qubit, rho_vec)
    #     res2 = Counter(dots_abs)
    # else:
    #     res2 = fast_compute_abs_dot(n_qubit, rho_vec)

    # sum_ = 0
    # for key, value in res2.items():
    #     sum_ += int(key) * value
    # print(f"sum: {sum_}")
    # if n_qubit <= 5:
    #     assert get_actual_Amat(n_qubit)[rho_vec > 0].count_nonzero() == sum_
    # assert sum_ == (
    #     (2**n_qubit - 1) * total_stabilizer_group_size(n_qubit) / (2**n_qubit + 1)
    # ) + total_stabilizer_group_size(n_qubit)

    df1[n_qubit] = list(
        map(format_int, [n_qubit, *[res1[i] for i in [0, 1, 2, 4, 8, 16, 32, 64]]])
    )
print("done")

df1.T.to_csv(
    os.path.join(os.path.dirname(__file__), "dot_with_col.csv"),
    index=False,
    header=True,
)
