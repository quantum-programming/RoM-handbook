import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import seaborn as sns
from _common import make_custom_cmap

from exputils.actual_Amat import get_actual_Amat
from exputils.stabilizer_group import idx_to_pauli_str

sns.set_theme("paper")
cmap = make_custom_cmap()


def plot_walsh():
    Amat = (
        scipy.sparse.kron(get_actual_Amat(1), get_actual_Amat(1))
        .toarray()
        .astype(np.float64)
    )
    Amat[Amat == 0] = np.nan
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(Amat, cbar=False, cmap=cmap, ax=ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks(
        0.5 + np.arange(16),
        [idx_to_pauli_str(2, i) for i in range(16)],
        fontsize=18,
        rotation=0,
        fontfamily="monospace",
    )

    plt.savefig(
        os.path.join(os.path.dirname(__file__), f"Amat_11.svg"),
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=500,
    )
    plt.clf()


if __name__ == "__main__":
    plot_walsh()
