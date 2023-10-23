import os

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns
from _common import make_custom_cmap

from exputils.actual_Amat import get_actual_Amat
from exputils.stabilizer_group import idx_to_pauli_str

sns.set_theme("paper")
cmap = make_custom_cmap()


def plot_Amat():
    A1 = get_actual_Amat(1).toarray().astype(np.float64)
    A1[A1 == 0] = np.nan
    A2 = get_actual_Amat(2).toarray().astype(np.float64)
    A2[A2 == 0] = np.nan

    grid_kw = dict(
        width_ratios=[A1.shape[1] / A1.shape[0], A2.shape[1] / A2.shape[0]], wspace=0.1
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw=grid_kw, figsize=(25, 5))

    sns.heatmap(A1, cbar=False, cmap=cmap, ax=ax1)
    sns.heatmap(A2, cbar=False, cmap=cmap, ax=ax2)
    ax1.set_aspect("equal", adjustable="box")
    ax2.set_aspect("equal", adjustable="box")
    ax1.set_title("n = 1", fontsize=25)
    ax2.set_title("n = 2", fontsize=25)
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax1.set_yticks(
        0.5 + np.arange(4),
        [idx_to_pauli_str(1, i) for i in range(4)],
        fontsize=18,
        rotation=0,
        fontfamily="monospace",
    )
    ax2.set_yticks(
        0.5 + np.arange(16),
        [idx_to_pauli_str(2, i) for i in range(16)],
        fontsize=18,
        rotation=0,
        fontfamily="monospace",
    )

    values = [+1, -1]
    colors = [cmap(255), cmap(0)]
    patches = [
        matplotlib.patches.Patch(color=colors[i], label="+1" if i == 0 else "-1")
        for i in range(len(values))
    ]

    fig.legend(
        handles=patches,
        bbox_to_anchor=(0.5, 0.05),
        loc="center",
        ncol=2,
        prop={"family": "monospace", "size": 20},
    )

    plt.savefig(
        os.path.join(os.path.dirname(__file__), "Amat.png"),
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=500,
    )
    plt.clf()


if __name__ == "__main__":
    plot_Amat()
    # adjust_images()
