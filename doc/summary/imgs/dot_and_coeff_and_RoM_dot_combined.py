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

sns.set_theme("paper")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rc = {
    "mathtext.fontset": "stix",
    "font.size": 20,
    "font.family": "Times New Roman",
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "text.usetex": True,
    "text.latex.preamble": "\\usepackage{amsmath}\n\\usepackage{bm}",
}
plt.rcParams.update(rc)


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


def visualize_performance(n_qubit, maxK, kind, ax: plt.Axes):
    global topK_Amat, cover_Amat, log_file_name

    log_file_name = os.path.join(
        os.path.dirname(__file__), f"result_RoM_dot_{kind}_{n_qubit}_data.txt"
    )

    cover_Amat = generators_to_Amat(n_qubit, make_cover_info(n_qubit)[0])

    seed = 0
    rho_vec = make_rho_vec(kind, n_qubit, seed)

    topK_Amat = get_topK_botK_Amat(
        n_qubit, rho_vec, maxK, is_dual=False, is_random=False, verbose=True
    )

    RoM_exact = calculate_RoM_actual(n_qubit, rho_vec, method="gurobi")[0]
    with open(log_file_name, mode="a") as f:
        print(f"{RoM_exact=}", file=f)
    if n_qubit <= 4:
        actual = calculate_RoM_actual(n_qubit, rho_vec, method="gurobi")[0]
        print(RoM_exact, actual)
        assert np.isclose(RoM_exact, actual)

    methods = ["Random", "Overlap", "Exact RoM"]
    functions = [
        calculate_RoM_random,
        wrapper_of_calculate_RoM_dot,
        lambda *a, **k: (RoM_exact, None),
    ]
    plot_colors = [colors[seed], colors[seed], "black"]
    markers = ["o", "o", None]
    alphas = [0.7, 1, 0.7]
    for p_id, (label, func, color, marker, alpha) in enumerate(
        zip(methods, functions, plot_colors, markers, alphas)
    ):
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

        sns.lineplot(
            data=pd.DataFrame({"K": np.linspace(0.1, 1.0, 10) * maxK, "RoM": RoMs}),
            x="K",
            y="RoM",
            color=color,
            label=label,
            alpha=alpha,
            marker=marker,
            linestyle=["-.", "-", "--"][p_id],
            zorder=1 if label == "Exact RoM" else 2,
            ax=ax,
        )

    ax.set_xlabel("$K$", fontsize=25)
    ax.set_ylabel(r"$\hat{\mathcal{R}}_0(\rho)$", fontsize=25)
    ax.set_title("(b)", fontsize=25, x=-0.07, y=1.05)


def visualize_multiple_Amat(n_qubit: int, seed: int, ax: plt.Axes):
    Amat = get_actual_Amat(n_qubit)
    rho = make_random_quantum_state("mixed", n_qubit, seed=seed, check=True)
    dot_product = (rho @ Amat).reshape(-1)
    RoM, coeff = calculate_RoM_custom(Amat, rho, method="gurobi")
    print(f"{RoM=}")
    sns.scatterplot(
        x=dot_product,
        y=coeff,
        marker="x",
        s=100,  # marker size
        linewidth=2,  # marker edge width
        ax=ax,
    )
    ax.set_xlabel(r"Unnormalized Overlap $\mathrm{Tr}[\rho\sigma_j]$", fontsize=25)
    ax.set_ylabel(r"Weight $x_j$", fontsize=25)
    ax.set_title("(a)", fontsize=25, x=-0.15, y=1.05)


if __name__ == "__main__":
    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    visualize_multiple_Amat(4, 0, ax1)
    ax2 = fig.add_subplot(1, 2, 2)
    visualize_performance(4, 0.1, "mixed", ax2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.dirname(__file__), f"dot_and_coeff_and_RoM_dot_combined.png"
        ),
        dpi=500,
    )
