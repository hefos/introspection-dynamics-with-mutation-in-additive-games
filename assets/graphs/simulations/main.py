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


def pgg_per_player_r(state, r_vector, contribution_vector, **kwargs):
    N = len(state)
    total = float(np.dot(contribution_vector, state))
    return np.array([
        r_vector[i] / N * total - contribution_vector[i] * float(state[i])
        for i in range(N)
    ], dtype=float)


def generate_stationary_table(out_path):
    N = 3
    alphas = np.ones(N)
    r_vector = np.array([1.0, 3.0, 9.0])   # r1 < N = r2 < r3
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

    # Formula p_i values: p_i = phi_i(alpha_i(1 - r_i/N)) * (1 - mu_i0 - mu_i1) + mu_i0
    p = np.array([
        1.0 / (1.0 + np.exp(beta * alphas[i] * (1.0 - r_vector[i] / N))) * (1 - 2 * mu) + mu
        for i in range(N)
    ])

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
        pi_formula = float(np.prod([
            p[i] ** int(state[i]) * (1 - p[i]) ** (1 - int(state[i]))
            for i in range(N)
        ]))
        lines.append(rf"{label} & {pi_formula:.5f} & {pi_exact:.5f} \\")
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
    Line2D([0], [0], marker="o", color="w", markerfacecolor="orange",
           markersize=8, label=r"exact $p_C$ (linear system)"),
    Line2D([0], [0], color="C0", linewidth=2,
           label=r"formula $p_C$ (Corollary~1)"),
    Patch(facecolor="C0", alpha=0.5, label=r"simulated $p_C$"),
]

fig.legend(handles=handles, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.01))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(file_path.parent / "main.pdf", bbox_inches="tight")

tex_dir = file_path.parents[3] / "tex"
generate_stationary_table(tex_dir / "stationary_table.tex")
