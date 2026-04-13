import ludics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

file_path = pathlib.Path(__file__)
assets_path = file_path.parents[2]


df_sim = pd.read_csv(assets_path / "data/simulations/r_gt_N/main.csv")
df_theoretic = pd.read_csv(assets_path / "data/theoretic_results/r_gt_N/main.csv")
df_new = pd.read_csv(assets_path / "data/new_formula/r_gt_N/main.csv")
df_sim["N"] = df_sim["N"].astype(str) 
df_new["N"] = df_new["N"].astype(str)
df_theoretic["N"] = df_theoretic["N"].astype(str)
fig, axes = plt.subplots(1,2, figsize=(14, 6))

sns.scatterplot(
    data=df_theoretic,
    x="N", y="p_C",
    ax=axes[0],
    marker='o',
    color='orange'
)

sns.lineplot(
    data=df_new,
    x="N", y="p_C",
    errorbar=None,
    ax=axes[0],
    markers='o',
)

sns.violinplot(
    data=df_sim,
    x="N", y="p_C",
    inner=None,
    ax=axes[0],
    facecolor='C0'
)

axes[0].set_title(r"$r > N$")
axes[0].set_xlabel(r"$N$")
axes[0].set_ylabel(r"$p_C$")

df_sim = pd.read_csv(assets_path / "data/simulations/r_lt_N/main.csv")
df_theoretic = pd.read_csv(assets_path / "data/theoretic_results/r_lt_N/main.csv")
df_new = pd.read_csv(assets_path / "data/new_formula/r_lt_N/main.csv")

df_sim["N"] = df_sim["N"].astype(str) 
df_new["N"] = df_new["N"].astype(str)
df_theoretic["N"] = df_theoretic["N"].astype(str)
sns.scatterplot(
    data=df_theoretic,
    x="N", y="p_C",
    ax=axes[1],
    marker='o',
    color='orange'
)

sns.lineplot(
    data=df_new,
    x="N", y="p_C",
    errorbar=None,
    ax=axes[1],
)

sns.violinplot(
    data=df_sim,
    x="N", y="p_C",
    inner=None,
    ax=axes[1],
    color='C0'
)

axes[1].set_title(r"$r < N$")
axes[1].set_xlabel(r"$N$")
axes[1].set_ylabel(r"$p_C$")

for ax in axes:
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    
    new_labels = []
    for lab in labels:
        try:
            n = int(lab)
            if n == 1  or n % 11 == 0:
                new_labels.append(lab)
            else:
                new_labels.append("")
        except ValueError:
            new_labels.append("")
    
    ax.set_xticklabels(new_labels)

for ax in axes:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='orange', markersize=8,
           label=r'actual $p_C$'),

    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='blue', markersize=8,
           label=r'theoretic $p_C$'),

    Patch(facecolor='C0', alpha=0.5,
          label=r'simulated $p_C$')
]

fig.legend(handles=handles, loc='upper center', ncol=3)

plt.tight_layout()
plt.legend()
plt.savefig(file_path.parent/"main.pdf")