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


def load_data(case):
    sim = pd.read_csv(assets_path / f"data/simulations/{case}/main.csv")
    theoretic = pd.read_csv(assets_path / f"data/theoretic_results/{case}/main.csv")
    new = pd.read_csv(assets_path / f"data/new_formula/{case}/main.csv")
    for df in (sim, theoretic, new):
        df["N"] = df["N"].astype(str)
    return sim, theoretic, new


def plot_case(ax, sim, theoretic, new):
    sns.scatterplot(data=theoretic, x="N", y="p_C", ax=ax,
                    marker="o", color="orange", zorder=3)
    sns.lineplot(data=new, x="N", y="p_C", errorbar=None, ax=ax, zorder=2)
    sns.violinplot(data=sim, x="N", y="p_C", inner=None, ax=ax,
                   facecolor="C0", zorder=1)

    labels = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticklabels([
        lab if (lab != "" and (int(lab) == 1 or int(lab) % 11 == 0)) else ""
        for lab in labels
    ])
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$p_C$")
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sim, theoretic, new = load_data("r_gt_N")
plot_case(axes[0], sim, theoretic, new)
axes[0].set_title(r"$r > N$")

sim, theoretic, new = load_data("r_lt_N")
plot_case(axes[1], sim, theoretic, new)
axes[1].set_title(r"$r < N$")

handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="orange",
           markersize=8, label=r"exact $p_C$ (linear system)"),
    Line2D([0], [0], color="C0", linewidth=2,
           label=r"formula $p_C$ (Theorem 1)"),
    Patch(facecolor="C0", alpha=0.5, label=r"simulated $p_C$"),
]

fig.legend(handles=handles, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.01))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(file_path.parent / "main.pdf", bbox_inches="tight")
