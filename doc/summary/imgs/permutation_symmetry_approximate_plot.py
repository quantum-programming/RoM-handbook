import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

files = ["result_pure_cg.txt", "result_H_cg.txt", "result_mixed_cg.txt"]
labels = ["Pure", r"Magic $|H\rangle$", "Mixed"]
exact = 7

sns.set_theme("paper")
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

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)

colors = sns.color_palette("colorblind", 3)

for colorIdx, (file, label) in enumerate(zip(files, labels)):
    ns = []
    RoMs = []
    with open(file) as f:
        for line in f.readlines():
            n_str, RoM_str = line.split()
            ns.append(int(n_str))
            RoMs.append(float(RoM_str))

    small_ns = [n for n in ns if n <= exact]
    small_RoMs = [RoM for n, RoM in zip(ns, RoMs) if n <= exact]
    large_ns = [n for n in ns if n > exact]
    large_RoMs = [RoM for n, RoM in zip(ns, RoMs) if n > exact]

    sns.scatterplot(
        data=pd.DataFrame({"n": small_ns, "RoM": small_RoMs}),
        x="n",
        y="RoM",
        ax=ax,
        label=label + ", " + "exact",
        s=20,
        color=colors[colorIdx],
        marker="o",
    )
    sns.scatterplot(
        data=pd.DataFrame({"n": large_ns, "RoM": large_RoMs}),
        x="n",
        y="RoM",
        ax=ax,
        label=label + ", " + "approximate",
        s=25,
        color=colors[colorIdx],
        marker="X",
    )

ax.set_yscale("log")
ax.set_xlabel("$n$", fontsize=15)
ax.set_ylabel(r"$\mathcal{R}(\rho^{\otimes n})$ or  $R_n$", fontsize=15)
ax.set_ylim(ymin=0.9)
ax.set_xticks([1, 7, 12, 17, 21])
ax.set_yticks(
    np.concatenate((np.arange(1, 10), np.arange(10, 100, 10), np.arange(100, 201, 100)))
)
legend = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 0.95))
for i in range(6):
    legend.legendHandles[i]._sizes = [40]

plt.savefig(
    "permutation_symmetry_approximate.pdf",
    bbox_inches="tight",
)
