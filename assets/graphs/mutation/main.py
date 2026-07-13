import numpy as np
import matplotlib.pyplot as plt
import pathlib

file_path = pathlib.Path(__file__)


def p_i(alpha, r, N, beta, mu):
    s = beta * alpha * (1 - r / N)
    return (1 - 2 * mu) / (1 + np.exp(s)) + mu


def p_i_alternative(alpha, r, N, beta, nu):
    s = beta * alpha * (1 - r / N)
    fermi_value = 1 / (1 + np.exp(s))
    return ((1 - nu) * fermi_value + nu) / (1 + nu)


linestyles = ["-", "--", ":"]
beta_values = np.linspace(0, 6, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

for ax, N in zip(axes, [5, 200]):
    for mu, ls in zip([0.0, 0.1, 0.25], linestyles):
        ax.plot(
            beta_values,
            p_i(alpha=2, r=7, N=N, beta=beta_values, mu=mu),
            color="black",
            linestyle=ls,
            linewidth=2,
            label=rf"$\mu = {mu}$",
        )
    beta_dots = beta_values[20::40]
    for label, mu in zip(["alternative scheme", None], [0.1, 0.25]):
        nu = mu / (1 - mu)
        ax.scatter(
            beta_dots,
            p_i_alternative(alpha=2, r=7, N=N, beta=beta_dots, nu=nu),
            marker="x",
            color="black",
            zorder=3,
            label=label,
        )
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"$\beta_i$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

axes[0].set_ylabel(r"$p_i$")
axes[0].set_title(r"$N = 5$ ($r_i = 7 > N$)")
axes[1].set_title(r"$N = 200$ ($r_i = 7 < N$)")

plt.tight_layout()
plt.savefig(file_path.parents[3] / "tex" / "mutation.pdf", bbox_inches="tight")
