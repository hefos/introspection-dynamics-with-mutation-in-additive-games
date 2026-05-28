import ludics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase


class HandlerBoxplot(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        cx, cy = width / 2, height / 2
        box_h = height * 0.5
        box_w = width * 0.45
        wlen = height * 0.25

        box = Rectangle(
            (cx - box_w / 2, cy - box_h / 2),
            box_w,
            box_h,
            facecolor=orig_handle.get_facecolor(),
            edgecolor="black",
            linewidth=0.8,
            transform=trans,
        )
        median = Line2D(
            [cx - box_w / 2, cx + box_w / 2],
            [cy, cy],
            color="black",
            linewidth=1.0,
            transform=trans,
        )
        whisker_lo = Line2D(
            [cx, cx],
            [cy - box_h / 2, cy - box_h / 2 - wlen],
            color="black",
            linewidth=0.8,
            transform=trans,
        )
        whisker_hi = Line2D(
            [cx, cx],
            [cy + box_h / 2, cy + box_h / 2 + wlen],
            color="black",
            linewidth=0.8,
            transform=trans,
        )
        cap_lo = Line2D(
            [cx - box_w / 4, cx + box_w / 4],
            [cy - box_h / 2 - wlen, cy - box_h / 2 - wlen],
            color="black",
            linewidth=0.8,
            transform=trans,
        )
        cap_hi = Line2D(
            [cx - box_w / 4, cx + box_w / 4],
            [cy + box_h / 2 + wlen, cy + box_h / 2 + wlen],
            color="black",
            linewidth=0.8,
            transform=trans,
        )
        return [box, median, whisker_lo, whisker_hi, cap_lo, cap_hi]


file_path = pathlib.Path(__file__)
assets_path = file_path.parents[2]


def load_data(case):
    sim = pd.read_csv(assets_path / f"data/quick_simulations/{case}/main.csv")
    theoretic = pd.read_csv(assets_path / f"data/theoretic_results/{case}/main.csv")
    new = pd.read_csv(assets_path / f"data/new_formula/{case}/main.csv")
    for df in (sim, theoretic, new):
        df["N"] = df["N"].astype(str)
    return sim, theoretic, new


def plot_case(ax, sim, theoretic, new):
    sns.scatterplot(
        data=theoretic, x="N", y="p_C", ax=ax, marker="o", color="orange", zorder=3
    )
    sns.lineplot(data=new, x="N", y="p_C", errorbar=None, ax=ax, zorder=2)
    sns.boxplot(
        data=sim,
        x="N",
        y="p_C",
        ax=ax,
        color="C0",
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="C0", edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker=".", markersize=3, color="black", alpha=0.5),
        zorder=1,
    )

    labels = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticklabels(
        [
            lab if (lab != "" and (int(lab) == 1 or int(lab) % 11 == 0)) else ""
            for lab in labels
        ]
    )
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$p_C$")
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def pgg_per_player_r(state, r_vector, contribution_vector, **kwargs):
    N = len(state)
    shared_pool = float(np.dot(r_vector * contribution_vector, state)) / N
    return np.array(
        [shared_pool - contribution_vector[i] * float(state[i]) for i in range(N)],
        dtype=float,
    )


def generate_stationary_table(out_path):
    N = 3
    alphas = np.array([1.0, 2.0, 3.0])
    r_vector = np.array([1.0, 3.0, 9.0])
    beta = 2.0
    mu = 0.1

    state_space = ludics.get_state_space(N=N, k=2)
    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=pgg_per_player_r,
        compute_transition_probability=ludics.compute_introspection_transition_probability,
        individual_to_action_mutation_probability=np.full(shape=(N, 2), fill_value=mu),
        r_vector=r_vector,
        contribution_vector=alphas,
        number_of_strategies=2,
        choice_intensity=beta,
    )
    steady_state = ludics.compute_steady_state(transition_matrix)

    p = np.array(
        [
            1.0
            / (1.0 + np.exp(beta * alphas[i] * (1.0 - r_vector[i] / N)))
            * (1 - 2 * mu)
            + mu
            for i in range(N)
        ]
    )

    state_to_idx = {tuple(s): i for i, s in enumerate(state_space)}

    lines = []
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(
        r"State $\mathbf{a}$ & Formula $\pi_{\mathbf{a}}$ & Exact $\pi_{\mathbf{a}}$ \\"
    )
    lines.append(r"\midrule")
    for state in state_space:
        idx = state_to_idx[tuple(state)]
        label = "".join("C" if a else "D" for a in state)
        pi_exact = float(steady_state[idx])
        pi_formula = float(
            np.prod(
                [
                    p[i] ** int(state[i]) * (1 - p[i]) ** (1 - int(state[i]))
                    for i in range(N)
                ]
            )
        )
        lines.append(rf"{label} & {pi_formula:.4f} & {pi_exact:.4f} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sim, theoretic, new = load_data("r_gt_N")
plot_case(axes[0], sim, theoretic, new)
axes[0].set_title(r"$r > N$")

sim, theoretic, new = load_data("r_lt_N")
plot_case(axes[1], sim, theoretic, new)
axes[1].set_title(r"$r < N$")

handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="orange",
        markersize=8,
        label=r"exact $p_C$ (linear system)",
    ),
    Line2D([0], [0], color="C0", linewidth=2, label=r"formula $p_C$ (Corollary~1)"),
    Patch(facecolor="C0", edgecolor="black", label=r"simulated $p_C$"),
]

fig.legend(
    handles=handles,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.01),
    handler_map={handles[2]: HandlerBoxplot()},
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(file_path.parents[3] / "tex" / "simulations.pdf", bbox_inches="tight")

tex_dir = file_path.parents[3] / "tex"
generate_stationary_table(tex_dir / "stationary_table.tex")
