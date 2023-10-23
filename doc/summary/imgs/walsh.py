import os

import matplotlib.pyplot as plt
import seaborn as sns
from _common import make_custom_cmap
from exputils.math.fwht import sylvesters


sns.set_theme("paper")
cmap = make_custom_cmap()


def plot_walsh():
    for n in range(1, 4 + 1):
        W = sylvesters(n)
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(W, cbar=False, cmap=cmap, ax=ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"n = {n}", fontsize=25)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(
            os.path.join(os.path.dirname(__file__), f"walsh_{n}.svg"),
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=500,
        )
        plt.clf()


if __name__ == "__main__":
    plot_walsh()
