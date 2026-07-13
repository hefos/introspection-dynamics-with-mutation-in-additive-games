import numpy as np
import matplotlib.pyplot as plt
import pathlib

file_path = pathlib.Path(__file__)


def p_i(alpha, r, N, beta, mu):
    s = beta * alpha * (1 - r / N)
    return (1 - 2 * mu) / (1 + np.exp(s)) + mu


N = 5
standard_alpha = 2
standard_mu = 0.1
linestyles = ["-", "--", "-.", ":"]

beta_values = np.linspace(0, 2, 400)
alpha_values = np.linspace(0.1, 10, 400)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
for r_val, ls in zip([2, 5, 7], linestyles):
    ax.plot(
        beta_values,
        p_i(standard_alpha, r_val, N, beta_values, standard_mu),
        color="black",
        linestyle=ls,
        linewidth=2,
        label=rf"$r_i = {r_val}$",
    )
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel(r"$\beta_i$")
ax.set_ylabel(r"$p_i$")
ax.legend()

ax = axes[1]
for (r_val, beta_val), ls in zip([(7, 2), (7, 0.2), (2, 2), (2, 0.2)], linestyles):
    ax.plot(
        alpha_values,
        p_i(alpha_values, r_val, N, beta_val, standard_mu),
        color="black",
        linestyle=ls,
        linewidth=2,
        label=rf"$r_i = {r_val},\ \beta_i = {beta_val}$",
    )
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel(r"$\alpha_i$")
ax.set_ylabel(r"$p_i$")
ax.legend()

ax = axes[2]
mu_values = np.linspace(0, 0.5, 400)
for r_val, ls in zip([2, 5, 7], linestyles):
    ax.plot(
        mu_values,
        p_i(standard_alpha, r_val, N, beta=2, mu=mu_values),
        color="black",
        linestyle=ls,
        linewidth=2,
        label=rf"$r_i = {r_val}$",
    )
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$p_i$")
ax.legend()

plt.tight_layout()
plt.savefig(file_path.parents[3] / "tex" / "big_panel.pdf", bbox_inches="tight")
