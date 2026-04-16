import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pathlib

file_path = pathlib.Path(__file__)


def p_i(alpha, r, N, beta, mu):
    s = beta * alpha * (1 - r / N)
    return (1 - 2 * mu) / (1 + np.exp(s)) + mu


def dp_dmu(alpha, r, N, beta):
    """dp_i/dmu = 1 - 2*phi(beta*alpha*(1-r/N)).  Independent of mu."""
    s = beta * alpha * (1 - r / N)
    return 1 - 2 / (1 + np.exp(s))


def dp_dalpha(alpha, r, N, beta, mu):
    s = beta * alpha * (1 - r / N)
    e_s = np.exp(s)
    return (1 - 2 * mu) * (-beta * (1 - r / N)) * e_s / (1 + e_s) ** 2


def dp_dbeta(alpha, r, N, beta, mu):
    s = beta * alpha * (1 - r / N)
    e_s = np.exp(s)
    return (1 - 2 * mu) * (-alpha * (1 - r / N)) * e_s / (1 + e_s) ** 2


N = 5
standard_alpha = 2
standard_mu = 0.1
standard_beta = 0.5
linestyles = ["-", "--", "-.", ":"]

beta_values = np.linspace(0, 2, 400)
alpha_values = np.linspace(0.1, 10, 400)
r_values = np.linspace(1, 10, 400)
beta_fine = np.linspace(0.01, 2, 400)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

ax = axes[0, 0]
for r_val, ls in zip([2, 5, 7], linestyles):
    ax.plot(
        beta_values,
        p_i(standard_alpha, r_val, N, beta_values, standard_mu),
        color="black",
        linestyle=ls,
        linewidth=2,
        label=rf"$r = {r_val}$",
    )
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel(r"$\beta_i$")
ax.set_ylabel(r"$p_i$")
ax.legend()

ax = axes[0, 1]
for (r_val, beta_val), ls in zip([(7, 2), (7, 0.2), (2, 2), (2, 0.2)], linestyles):
    ax.plot(
        alpha_values,
        p_i(alpha_values, r_val, N, beta_val, standard_mu),
        color="black",
        linestyle=ls,
        linewidth=2,
        label=rf"$r = {r_val},\ \beta = {beta_val}$",
    )
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel(r"$\alpha_i$")
ax.set_ylabel(r"$p_i$")
ax.legend()

ax = axes[0, 2]
R, A = np.meshgrid(r_values, alpha_values)
Z = p_i(A, R, N, beta=0, mu=standard_mu)
im = ax.imshow(
    Z,
    extent=[r_values.min(), r_values.max(), alpha_values.min(), alpha_values.max()],
    origin="lower",
    aspect="auto",
    cmap="gray",
    vmin=0,
    vmax=1,
)
plt.colorbar(im, ax=ax, label=r"$p_i$")
ax.axvline(N, color="k", linewidth=1, linestyle="--", label=rf"$r_i = N = {N}$")
ax.set_xlabel(r"$r_i$")
ax.set_ylabel(r"$\alpha_i$")
ax.set_title(r"$p_i$ when $\beta_i = 0$" "\n" r"(neutral drift: $p_i = 1/2$)")
ax.legend(fontsize=8)


def diverging_heatmap(ax, Z, r_vals, y_vals, xlabel, ylabel, cblabel, N):
    vmax = np.abs(Z).max()
    vmin = -vmax
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(
        Z,
        extent=[r_vals.min(), r_vals.max(), y_vals.min(), y_vals.max()],
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
    )
    plt.colorbar(im, ax=ax, label=cblabel)
    ax.axvline(N, color="k", linewidth=1, linestyle="--", label=rf"$r_i = N = {N}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


ax = axes[1, 0]
R, A = np.meshgrid(r_values, alpha_values)
Z = dp_dmu(A, R, N, beta=standard_beta)
diverging_heatmap(
    ax,
    Z,
    r_values,
    alpha_values,
    xlabel=r"$r_i$",
    ylabel=r"$\alpha_i$",
    cblabel=r"$\partial p_i / \partial \mu$",
    N=N,
)
ax.set_title(
    r"$\partial p_i / \partial \mu$"
    "\n"
    rf"($\beta_i = {standard_beta}$;  $\mu$ does not appear in derivative)"
)

ax = axes[1, 1]
R, A = np.meshgrid(r_values, alpha_values)
Z = dp_dalpha(A, R, N, beta=standard_beta, mu=standard_mu)
diverging_heatmap(
    ax,
    Z,
    r_values,
    alpha_values,
    xlabel=r"$r_i$",
    ylabel=r"$\alpha_i$",
    cblabel=r"$\partial p_i / \partial \alpha_i$",
    N=N,
)
ax.set_title(
    r"$\partial p_i / \partial \alpha_i$"
    "\n"
    rf"($\beta_i = {standard_beta}$, $\mu_i = {standard_mu}$)"
)

ax = axes[1, 2]
R, B = np.meshgrid(r_values, beta_fine)
Z = dp_dbeta(standard_alpha, R, N, B, mu=standard_mu)
diverging_heatmap(
    ax,
    Z,
    r_values,
    beta_fine,
    xlabel=r"$r_i$",
    ylabel=r"$\beta_i$",
    cblabel=r"$\partial p_i / \partial \beta_i$",
    N=N,
)
ax.set_title(
    r"$\partial p_i / \partial \beta_i$"
    "\n"
    rf"($\alpha_i = {standard_alpha}$, $\mu_i = {standard_mu}$)"
)

# ---------------------------------------------------------------------------

plt.tight_layout()
plt.savefig(file_path.parent / "main.pdf", bbox_inches="tight")
