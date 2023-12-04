import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import seaborn as sns

from exputils.actual_Amat import generators_to_Amat, get_actual_Amat
from exputils.cover_states import make_cover_info
from exputils.dot.get_topK_botK_Amat import get_topK_botK_Amat
from exputils.RoM.actual import calculate_RoM_actual
from exputils.RoM.custom import calculate_RoM_custom
from exputils.RoM.dot import get_topK_indices
from exputils.RoM.random import calculate_RoM_random
from exputils.stabilizer_group import total_stabilizer_group_size
from exputils.state.random import make_random_quantum_state
from exputils.state.tensor import make_random_tensor_product_state


def make_rho_vec(rho_vec_method: str, n_qubit: int, seed: int):
    if rho_vec_method == "mixed":
        rho_vec = make_random_quantum_state(rho_vec_method, n_qubit, seed)
    elif rho_vec_method == "pure":
        rho_vec = make_random_quantum_state(rho_vec_method, n_qubit, seed)
    elif rho_vec_method == "tensor":
        rho_vec = make_random_tensor_product_state("mixed", n_qubit, seed)
    elif rho_vec_method in ["H", "F"]:
        rho_vec = make_random_tensor_product_state(rho_vec_method, n_qubit, seed)
    else:
        raise ValueError
    return rho_vec


topK_Amat = None
cover_Amat = None
log_file_name = None


def calculate_RoM_CG(
    n_qubit: int,
    _K: float,
    rho_vec: np.ndarray,
    rho_vec_method: str,
    method: str = "gurobi",
    verbose: bool = False,
    iter_max: int = 100,
    eps: float = 1e-5,
    discard_current_threshold: float = 0.9,
):
    global topK_Amat
    global log_file_name

    K = [1, 1, 1, 1, 0.1, 0.01, 0.001, 0.00001][n_qubit]
    assert K == _K

    if rho_vec_method == "tensor":
        current_Amat = get_actual_Amat(1)
        for _ in range(n_qubit - 1):
            current_Amat = scipy.sparse.kron(
                current_Amat, get_actual_Amat(1), format="csc"
            )
    else:
        assert topK_Amat is not None
        current_Amat = topK_Amat
        # current_Amat = get_topK_botK_Amat(n_qubit, rho_vec, K, verbose=False)

    RoM_hist = []
    violations_hist = []

    for it in range(iter_max):
        print(
            f"iteration: {it + 1} / {iter_max}, # of columns = {current_Amat.shape[1]}"
        )
        RoM, coeff, dual = calculate_RoM_custom(
            current_Amat,
            rho_vec,
            method=method,
            return_dual=True,
            crossover=False,
            presolve=False,
            verbose=verbose,
        )
        print(f"{RoM = }")
        with open(log_file_name, mode="a") as f:
            print(f"{RoM = }", file=f)

        RoM_hist.append(RoM)

        dual_Amat = get_topK_botK_Amat(n_qubit, dual, K, verbose=False)
        dual_dots = np.abs(dual.T @ dual_Amat)
        dual_violated_indices = dual_dots > 1 + eps
        violated_count = np.sum(dual_violated_indices)
        violations_hist.append(violated_count)
        print(
            "# of violations:",
            f"{violated_count}"
            if violated_count < dual_Amat.shape[1]
            else f"more than {violated_count}",
        )
        with open(log_file_name, mode="a") as f:
            print(
                "# of violations:",
                f"{violated_count}"
                if violated_count < dual_Amat.shape[1]
                else f"more than {violated_count}",
                file=f,
            )

        nonbasic_indices = np.abs(coeff) > eps
        critical_indices = np.abs(dual @ current_Amat) >= (
            discard_current_threshold - eps
        )
        remain_indices = np.logical_or(nonbasic_indices, critical_indices)
        current_Amat = current_Amat[:, remain_indices]

        if violated_count == 0:
            print("exact RoM found!")
            break
        else:
            indices = np.where(dual_violated_indices)[0]
            extra_Amat = dual_Amat[:, indices]

        current_Amat = scipy.sparse.hstack((current_Amat, extra_Amat))

    assert len(RoM_hist) == len(violations_hist)
    print(f"{RoM_hist=}")
    print(f"{violations_hist=}")

    return RoM


def wrapper_of_calculate_RoM_dot(
    n_qubit, rho_vec, K, method, kind, presolve, crossover, verbose
):
    global topK_Amat, cover_Amat
    assert topK_Amat is not None
    assert kind != "tensor" or cover_Amat is not None
    assert 0 < K <= 1
    size = int(round(total_stabilizer_group_size(n_qubit) * K))
    assert size <= topK_Amat.shape[1] + 100, f"{topK_Amat.shape[1]=}, {size=}"
    dots = rho_vec.T @ topK_Amat
    idxs = get_topK_indices(dots, min(1, size / topK_Amat.shape[1]))

    Amat = (
        scipy.sparse.hstack([topK_Amat[:, idxs], cover_Amat])
        if kind == "tensor"
        else topK_Amat[:, idxs]
    )
    return calculate_RoM_custom(
        Amat,
        rho_vec,
        method=method,
        presolve=presolve,
        crossover=crossover,
        verbose=verbose,
    )


def visualize_performance(n_qubit, maxK, kind):
    global topK_Amat, cover_Amat, log_file_name

    log_file_name = os.path.join(
        os.path.dirname(__file__), f"result_RoM_dot/{kind}_{n_qubit}_data.txt"
    )

    cover_Amat = generators_to_Amat(n_qubit, make_cover_info(n_qubit)[0])

    seed = 0
    rho_vec = make_rho_vec(kind, n_qubit, seed)

    topK_Amat = get_topK_botK_Amat(n_qubit, rho_vec, maxK, verbose=True)

    RoM_exact = calculate_RoM_CG(
        n_qubit,
        maxK,
        rho_vec,
        kind,
        verbose=True,
        discard_current_threshold=0.8 if n_qubit == 7 and kind == "pure" else 0.9,
    )
    with open(log_file_name, mode="a") as f:
        print(f"{RoM_exact=}", file=f)

    if n_qubit <= 4:
        actual = calculate_RoM_actual(n_qubit, rho_vec, method="gurobi")[0]
        print(RoM_exact, actual)
        assert np.isclose(RoM_exact, actual)

    plt.figure(figsize=(8, 5))
    methods = ["Random", "Inner Product", "Exact RoM"]
    functions = [
        calculate_RoM_random,
        wrapper_of_calculate_RoM_dot,
        lambda *a, **k: (RoM_exact, None),
    ]
    for label, func in zip(methods, functions):
        RoMs = []
        for K in np.linspace(0.1, 1.0, 10) * maxK:
            t0 = time.perf_counter()
            if label == "Random":
                RoM = func(
                    n_qubit,
                    rho_vec,
                    K,
                    method="gurobi",
                    verbose=n_qubit >= 6,
                    presolve=False,
                    crossover=False,
                )[0]
            else:
                RoM = func(
                    n_qubit,
                    rho_vec,
                    K,
                    kind=kind,
                    method="gurobi",
                    verbose=n_qubit >= 6,
                    presolve=False,
                    crossover=False,
                )[0]
            t1 = time.perf_counter()
            RoMs.append(RoM)
            print(f"{n_qubit=}, {seed=}, {K=:.5f}, {label=}, {RoM=} {t1-t0=:.5f}")
            with open(log_file_name, mode="a") as f:
                print(
                    f"{n_qubit=}, {seed=}, {K=:.5f}, {label=}, {RoM=} {t1-t0=:.5f}",
                    file=f,
                )

    print("done")


if __name__ == "__main__":
    visualize_performance(4, 0.1, "mixed")
    visualize_performance(4, 0.1, "pure")
    visualize_performance(4, 0.1, "tensor")
    visualize_performance(5, 0.01, "mixed")
    visualize_performance(5, 0.01, "pure")
    visualize_performance(5, 0.01, "tensor")
    visualize_performance(6, 0.001, "mixed")
    visualize_performance(6, 0.001, "pure")
    visualize_performance(6, 0.001, "tensor")

    # the following takes a long time
    # We run the following one by one with comment out and
    # saved the results and plotted them after all the runs are done.

    # visualize_performance(7, 0.00001, "mixed")
    # visualize_performance(7, 0.00001, "pure")
    # visualize_performance(7, 0.00001, "tensor")
