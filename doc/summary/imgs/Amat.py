import os

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from _common import make_custom_cmap
from PIL import Image

from exputils.actual_Amat import get_actual_Amat
from exputils.stabilizer_group import idx_to_pauli_str

sns.set_theme("paper")
cmap = make_custom_cmap()


def plot_Amat():
    for n_qubit in [1, 2]:
        A = get_actual_Amat(n_qubit).toarray().astype(np.float64)
        A[A == 0] = np.nan
        plt.figure(figsize=(15, 4))
        plt.gca().set_aspect("equal", adjustable="box")
        sns.heatmap(A, cbar=False, cmap=cmap)
        plt.title(f"n = {n_qubit}", fontsize=25)
        plt.yticks(
            0.5 + np.arange(4**n_qubit),
            [idx_to_pauli_str(2, i) for i in range(4**n_qubit)],
            fontsize=12 if n_qubit == 2 else 18,
            rotation=0,
            fontfamily="monospace",
        )
        plt.xticks([])
        if n_qubit == 1:
            values = [+1, -1]
            colors = [cmap(255), cmap(0)]
            patches = [
                matplotlib.patches.Patch(
                    color=colors[i], label="+1" if i == 0 else "-1"
                )
                for i in range(len(values))
            ]
            plt.legend(
                handles=patches,
                bbox_to_anchor=(1.5, 0.5),
                loc="center right",
                fontsize=27,
            )
            plt.savefig(os.path.join(os.path.dirname(__file__), "Amat_1.png"), dpi=500)
        else:
            plt.savefig(
                os.path.join(os.path.dirname(__file__), "Amat_2.png"),
                bbox_inches="tight",
                pad_inches=0.1,
                dpi=500,
            )
        plt.clf()


def adjust_images():
    A1 = Image.open(os.path.join(os.path.dirname(__file__), "Amat_1.png"))
    A2 = Image.open(os.path.join(os.path.dirname(__file__), "Amat_2.png"))
    A1_width = A1.size[0]
    width, height = A2.size
    result = Image.new(A1.mode, (width, 2 * height + 100), color=(255, 255, 255))
    print(width, height, A1_width)
    result.paste(A1, ((width - A1_width) // 2, 0))
    result.paste(A2, (0, height + 100))
    result.save(
        os.path.join(os.path.dirname(__file__), "Amat.png"),
        bbox_inches="tight",
        pad_inches=0.1,
    )


if __name__ == "__main__":
    plot_Amat()
    adjust_images()
