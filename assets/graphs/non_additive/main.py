import ludics
import numpy as np
import matplotlib.pyplot as plt
import pathlib

file_path = pathlib.Path(__file__)

N = 8
r = 4.0
linestyles = ["-", "--", "-."]
w_values = [0.7, 1.0, 1.4]
unit_alphas = np.ones(N)
heterogeneous_alphas = np.arange(1, N + 1, dtype=float)


def nonlinear_pgg(state, r, w, alphas, **kwargs):
    total_contribution = float(np.dot(alphas, state))
    if w == 1.0:
        pool = r * total_contribution / N
    else:
        pool = r * (w**total_contribution - 1) / ((w - 1) * N)
    return np.array(
        [pool - alphas[i] * float(state[i]) for i in range(N)]
    )


def steady_state_and_space(r, w, beta, mu, alphas):
    state_space = ludics.get_state_space(N=N, k=2)
    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=nonlinear_pgg,
        compute_transition_probability=ludics.compute_introspection_transition_probability,
        individual_to_action_mutation_probability=np.full(shape=(N, 2), fill_value=mu),
        r=r,
        w=w,
        alphas=alphas,
        number_of_strategies=2,
        choice_intensity=beta,
    )
    return ludics.compute_steady_state(transition_matrix), state_space


def exact_p_C(r, w, beta, mu):
    steady_state, state_space = steady_state_and_space(
        r=r, w=w, beta=beta, mu=mu, alphas=unit_alphas
    )
    coop_fraction = np.array([np.mean(state) for state in state_space])
    return float(steady_state @ coop_fraction)


def correlation_matrix(r, w, beta, mu, alphas):
    steady_state, state_space = steady_state_and_space(
        r=r, w=w, beta=beta, mu=mu, alphas=alphas
    )
    actions = np.array(
        [[float(state[i]) for i in range(N)] for state in state_space]
    )
    means = steady_state @ actions
    covariance = (actions * steady_state[:, None]).T @ actions - np.outer(
        means, means
    )
    stddev = np.sqrt(means * (1 - means))
    return covariance / np.outer(stddev, stddev)


def additive_p_C(beta, mu):
    return (1 - 2 * mu) / (1 + np.exp(beta * (1 - r / N))) + mu


fig, axes = plt.subplots(2, 3, figsize=(15, 9))

mu_values = np.concatenate(
    [np.linspace(0.002, 0.1, 20), np.linspace(0.12, 0.5, 14)]
)

ax = axes[0, 0]
beta_fixed = 5.0
for w, ls in zip(w_values, linestyles):
    ax.plot(
        mu_values,
        [exact_p_C(r=r, w=w, beta=beta_fixed, mu=mu) for mu in mu_values],
        color="black",
        linestyle=ls,
        linewidth=2,
        label=rf"$w = {w}$",
    )
mu_dots = mu_values[::4]
ax.scatter(
    mu_dots,
    [additive_p_C(beta=beta_fixed, mu=mu) for mu in mu_dots],
    marker="x",
    color="black",
    zorder=3,
    label=r"formula ($w = 1$)",
)
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$p_C$")
ax.set_title(rf"$r = {r:.0f}$, $\beta = {beta_fixed:.0f}$")
ax.legend()

ax = axes[0, 1]
beta_values = np.linspace(0, 10, 34)
mu_fixed = 0.05
for w, ls in zip(w_values, linestyles):
    ax.plot(
        beta_values,
        [exact_p_C(r=r, w=w, beta=beta, mu=mu_fixed) for beta in beta_values],
        color="black",
        linestyle=ls,
        linewidth=2,
        label=rf"$w = {w}$",
    )
beta_dots = beta_values[::4]
ax.scatter(
    beta_dots,
    [additive_p_C(beta=beta, mu=mu_fixed) for beta in beta_dots],
    marker="x",
    color="black",
    zorder=3,
    label=r"formula ($w = 1$)",
)
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$p_C$")
ax.set_title(rf"$r = {r:.0f}$, $\mu = {mu_fixed}$")
ax.legend()

ax = axes[0, 2]
bistable_r, bistable_w = 2.0, 1.5
for beta, ls in zip([5.0, 10.0, 20.0], linestyles):
    ax.plot(
        mu_values,
        [
            exact_p_C(r=bistable_r, w=bistable_w, beta=beta, mu=mu)
            for mu in mu_values
        ],
        color="black",
        linestyle=ls,
        linewidth=2,
        label=rf"$\beta = {beta:.0f}$",
    )
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$p_C$")
ax.set_title(rf"$r = {bistable_r:.0f}$, $w = {bistable_w}$")
ax.legend()

correlation_w_values = [0.95, 1.0, 1.05]
heterogeneous_beta = 0.5
for column, w in enumerate(correlation_w_values):
    ax = axes[1, column]
    corr = correlation_matrix(
        r=r,
        w=w,
        beta=heterogeneous_beta,
        mu=mu_fixed,
        alphas=heterogeneous_alphas,
    )
    np.fill_diagonal(corr, np.nan)
    scale = max(0.03, np.nanmax(np.abs(corr)))
    im = ax.imshow(
        corr, origin="lower", cmap="RdBu_r", vmin=-scale, vmax=scale
    )
    fig.colorbar(im, ax=ax, label=r"$\mathrm{corr}(a_i, a_j)$")
    ax.set_xticks(range(N), labels=range(1, N + 1))
    ax.set_yticks(range(N), labels=range(1, N + 1))
    ax.set_xlabel(r"player $j$ ($\alpha_j = j$)")
    ax.set_title(rf"$w = {w}$")
axes[1, 0].set_ylabel(r"player $i$ ($\alpha_i = i$)")

plt.tight_layout()
plt.savefig(file_path.parents[3] / "tex" / "non_additive.pdf", bbox_inches="tight")
