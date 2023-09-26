import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from exputils.RoM.actual import calculate_RoM_actual
from exputils.RoM.dot import calculate_RoM_dot
from exputils.RoM.random import calculate_RoM_random
from exputils.state.random import make_random_quantum_state
from exputils.state.tensor import make_random_tensor_product_state

sns.set_theme("paper")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
sns.set(font_scale=1.5)


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


def visualize_performance():
    n_qubit = 4
    Ks = (np.linspace(0.1, 1, 10) * (0.1 ** (n_qubit - 3))).tolist()
    seed = 0
    rho_vec = make_rho_vec("mixed", n_qubit, seed)
    RoM_exact = calculate_RoM_actual(n_qubit, rho_vec, method="gurobi")[0]

    plt.figure(figsize=(8, 5))
    methods = ["Random", "Inner Product", "Exact RoM"]
    funcs = [calculate_RoM_random, calculate_RoM_dot, lambda *a, **k: (RoM_exact, None)]
    plot_colors = [colors[seed], colors[seed], "black"]
    markers = ["o", "o", None]
    alphas = [0.3, 1, 0.5]
    for label, func, color, marker, alpha in zip(
        methods, funcs, plot_colors, markers, alphas
    ):
        RoMs = []
        for K in Ks:
            t0 = time.perf_counter()
            RoM = func(n_qubit, rho_vec, K, method="gurobi")[0]
            t1 = time.perf_counter()
            RoMs.append(RoM)
            print(f"{n_qubit=}, {seed=}, {K=:.5f}, {label=}, {RoM=} {t1-t0=:.5f}")

        sns.lineplot(
            data=pd.DataFrame({"K": Ks, "RoM": RoMs}),
            x="K",
            y="RoM",
            color=color,
            label=label,
            alpha=alpha,
            marker=marker,
            linestyle="--" if label == "Exact RoM" else "-",
            zorder=1 if label == "Exact RoM" else 2,
        )

    plt.title(f"n = {n_qubit}", fontsize=20)
    plt.xlabel("K", fontsize=20)
    plt.ylabel("RoM", fontsize=20)

    plt.savefig(
        os.path.join(os.path.dirname(__file__), f"RoM_dot_mixed_{n_qubit}.pdf"),
        bbox_inches="tight",
    )
    print("done")


if __name__ == "__main__":
    visualize_performance()
