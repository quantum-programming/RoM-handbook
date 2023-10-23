import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
import seaborn as sns
import pandas as pd

files = ["result_mixed_cg.txt", "result_pure_cg.txt", "result_H_cg.txt"]
labels = ["Mixed", "Pure", r"Magic $|{H}\rangle$"]
exact = 7

sns.set_theme("paper")
sns.set(font_scale=1.2)

dfs = []
for file, label in zip(files, labels):
    with open(file) as f:
        lines = f.readlines()
    ns = []
    RoMs = []
    for line in lines:
        n_str, RoM_str = line.split()
        ns.append(int(n_str))
        RoMs.append(float(RoM_str))
    ns = ns
    types = [label] * len(ns)
    isexact = ["Exact" if n <= exact else "Approximate" for n in ns]
    dfs.append(
        pd.DataFrame({"n": ns, "RoM": RoMs, "state": types, "solution": isexact})
    )

df = pd.concat(dfs)

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)

sns.scatterplot(data=df, x="n", y="RoM", hue="state", style="solution")

ax.set_yscale("log")
ax.set_xlabel("n", fontsize=15)
ax.set_ylabel("RoM", fontsize=15)
ax.set_ylim(ymin=0.9)
ax.set_yticks(
    np.concatenate((np.arange(1, 10), np.arange(10, 100, 10), np.arange(100, 201, 100)))
)

handles, labels = ax.get_legend_handles_labels()
handles.insert(4, handles[0])
labels.insert(4, "")
ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
plt.setp(ax.get_legend().get_texts(), fontsize="12")

plt.savefig(
    "permutation_symmetry_approximate.pdf",
    bbox_inches="tight",
)
