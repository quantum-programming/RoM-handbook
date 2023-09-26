import os
import seaborn as sns
import matplotlib.pyplot as plt
from exputils.RoM.custom import calculate_RoM_custom
from exputils.actual_Amat import get_actual_Amat
from exputils.state.random import make_random_quantum_state


sns.set_theme("paper")
sns.set(font_scale=1.5)


def visualize_multiple_Amat(n_qubit: int, seed: int):
    plt.figure(figsize=(8, 5))
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
    )
    plt.title(f"n = {n_qubit}", fontsize=20)
    plt.xlabel("inner product with each column of A matrix", fontsize=20)
    plt.ylabel("optimal quasi-probability", fontsize=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), f"dot_and_coeff_{n_qubit}.png"),
        dpi=500,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    visualize_multiple_Amat(4, seed=0)
