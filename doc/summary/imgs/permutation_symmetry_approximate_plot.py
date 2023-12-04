import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

files = ["result_mixed_cg.txt", "result_pure_cg.txt", "result_H_cg.txt"]
labels = ["Mixed", "Pure", r"Magic $|H\rangle$"]
exact = 7

sns.set_theme("paper")
rc = {"mathtext.fontset": "stix"}
plt.rcParams.update(rc)
sns.set(font_scale=1.2, font="Times New Roman")

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)

colors = sns.color_palette("colorblind", 3)

for file, label in zip(files, labels):
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

    if label == "Mixed":
        colorIdx = 0
    elif label == "Pure":
        colorIdx = 1
    else:
        colorIdx = 2

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
ax.set_ylabel("RoM", fontsize=15)
ax.set_ylim(ymin=0.9)
ax.set_yticks(
    np.concatenate((np.arange(1, 10), np.arange(10, 100, 10), np.arange(100, 201, 100)))
)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))

plt.savefig(
    "permutation_symmetry_approximate.pdf",
    bbox_inches="tight",
)
