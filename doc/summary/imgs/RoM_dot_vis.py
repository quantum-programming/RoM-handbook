import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme("paper")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
rc = {"mathtext.fontset": "stix"}
rc = {
    "mathtext.fontset": "stix",
    "font.size": 25,
    "font.family": "Times New Roman",
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 23,
    "text.usetex": True,
    "text.latex.preamble": "\\usepackage{amsmath}\n\\usepackage{bm}",
}
plt.rcParams.update(rc)


def load_result(n: int, kind: str):
    RoM_exact = -1
    RoM_random = []
    RoM_overlap = []
    with open(f"doc/summary/imgs/result_RoM_dot/{kind}_{n}_data.txt") as f:
        for line in f.readlines():
            if line.startswith("RoM_exact"):
                RoM_exact = float(line[line.index("=") + 1 :])
            if line.startswith(f"n_qubit={n}"):
                if "Random" in line:
                    string = line.split()[-2]
                    RoM = float(string[string.index("=") + 1 :])
                    RoM_random.append(RoM)
                elif "Inner Product" in line:
                    string = line.split()[-2]
                    RoM = float(string[string.index("=") + 1 :])
                    RoM_overlap.append(RoM)
                else:
                    assert "Exact" in line
    return RoM_random, RoM_overlap, [RoM_exact] * 10


def main():
    seed = 0

    plot_colors = [colors[seed], colors[seed], "black"]
    markers = ["o", "o", None]
    labels = ["Random", "Overlap", "Exact RoM"]
    alphas = [0.7, 1, 0.7]

    fig = plt.figure(figsize=(8 * 3, 5 * 4 + 0.5 * 3))

    for n_id, n_qubit in enumerate([4, 5, 6, 7]):
        for k_id, kind in enumerate(["mixed", "pure", "tensor"]):
            RoMs_tuple = load_result(n_qubit, kind)
            ax = fig.add_subplot(4, 3, n_id * 3 + k_id + 1)
            maxK = [0.1, 0.01, 0.001, 0.00001][n_id]
            for p_id, (color, marker, label, alpha, RoMs) in enumerate(
                zip(plot_colors, markers, labels, alphas, RoMs_tuple)
            ):
                sns.lineplot(
                    data=pd.DataFrame(
                        {"K": np.linspace(0.1, 1.0, 10) * maxK, "RoM": RoMs}
                    ),
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

            ax.set_xlabel("$K$", fontsize=30)
            ax.set_ylabel(r"$\hat{\mathcal{R}}_0(\rho)$", fontsize=30)

            from matplotlib.ticker import FormatStrFormatter

            if np.max(RoMs_tuple[0]) < 10:
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

            kind_title = "Tensor Product" if kind == "tensor" else kind.capitalize()
            ax.set_title(
                f"{kind_title} ($n = {n_qubit}$)",
                fontsize=30,
                y=1.05,
            )
            ax.text(
                -0.145,
                1.05,
                f"({chr(97 + n_id * 3 + k_id)})",
                transform=ax.transAxes,
                size=30,
            )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.savefig("doc/summary/imgs/RoM_dot_vis.pdf")


if __name__ == "__main__":
    main()
