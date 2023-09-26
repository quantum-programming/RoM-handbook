import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file = "result_mixed_cg.txt"
exact = 7

sns.set_theme("paper")
sns.set(font_scale=1.5)


with open(file) as f:
    lines = f.readlines()

ns = []
RoMs = []
for line in lines:
    n_str, RoM_str = line.split()
    ns.append(int(n_str))
    RoMs.append(float(RoM_str))


ns_exact = ns[:exact]
ns_approx = ns[exact:]
RoMs_exact = RoMs[:exact]
RoMs_approx = RoMs[exact:]

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ns_list = [ns_exact, ns_approx]
RoMs_list = [RoMs_exact, RoMs_approx]
labels = ["Exact", "Approximate"]
for ns, RoMs, label in zip(ns_list, RoMs_list, labels):
    sns.scatterplot(x=ns, y=RoMs, label=label, linewidth=0, marker="o", ax=ax)

ax.set_yscale("log")
ax.set_xlabel("n", fontsize=20)
ax.set_ylabel("RoM", fontsize=20)
ax.set_ylim(ymin=0.9, ymax=35)
ax.set_yticks(np.concatenate((np.arange(1, 10), np.arange(10, 40, 10))))

plt.savefig(
    "permutation_symmetry_approximate.pdf",
    bbox_inches="tight",
)
