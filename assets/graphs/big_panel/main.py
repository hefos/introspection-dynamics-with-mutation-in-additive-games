import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

black_grey = LinearSegmentedColormap.from_list(
    "black_grey",
    ["black", "dimgray", "lightgray"]
)
file_path = pathlib.Path(__file__)

def get_individual_player_val(alpha, r, N, beta, mu):
    numerator = 1 - (2 * mu)
    denominator = 1 + np.exp(beta * (alpha * (1 - (r / N))))
    return (numerator / denominator) + mu

fig = plt.figure(figsize=(12, 9))
gs = GridSpec(2, 3, figure=fig)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0], projection='3d')
ax4 = fig.add_subplot(gs[1, 1], projection='3d')
ax5 = fig.add_subplot(gs[1, 2], projection='3d')

sns.set_style("whitegrid")

linestyles = ['-', '--', '-.', ':']

N = 5
standard_alpha = 2
standard_mu = 0.1

beta_values = np.linspace(0, 2, 2000)

beta_values_r_lt = np.array([
    get_individual_player_val(standard_alpha, 2, N, beta, standard_mu)
    for beta in beta_values
])

beta_values_r_eq = np.array([
    get_individual_player_val(standard_alpha, 5, N, beta, standard_mu)
    for beta in beta_values
])

beta_values_r_gt = np.array([
    get_individual_player_val(standard_alpha, 7, N, beta, standard_mu)
    for beta in beta_values
])

sns.lineplot(x=beta_values, y=beta_values_r_lt, ax=ax0,
             color='black', linestyle=linestyles[0], label=r'$r=2$')

sns.lineplot(x=beta_values, y=beta_values_r_eq, ax=ax0,
             color='black', linestyle=linestyles[1], label=r'$r=5$')

sns.lineplot(x=beta_values, y=beta_values_r_gt, ax=ax0,
             color='black', linestyle=linestyles[2], label=r'$r=7$')

ax0.set_xlabel(r'$\beta$')
ax0.set_ylabel(r'$p_i$')
ax0.legend()

alpha_values = np.linspace(0.1, 10, 2000)

alpha_values_r_gt_beta_high = np.array([
    get_individual_player_val(alpha, 7, N, 2, standard_mu)
    for alpha in alpha_values
])

alpha_values_r_gt_beta_low = np.array([
    get_individual_player_val(alpha, 7, N, 0.2, standard_mu)
    for alpha in alpha_values
])

alpha_values_r_lt_beta_high = np.array([
    get_individual_player_val(alpha, 2, N, 2, standard_mu)
    for alpha in alpha_values
])

alpha_values_r_lt_beta_low = np.array([
    get_individual_player_val(alpha, 2, N, 0.2, standard_mu)
    for alpha in alpha_values
])

sns.lineplot(x=alpha_values, y=alpha_values_r_gt_beta_high, ax=ax1,
             color='black', linestyle=linestyles[0],
             label=r'$r=7,\beta=2$')

sns.lineplot(x=alpha_values, y=alpha_values_r_gt_beta_low, ax=ax1,
             color='black', linestyle=linestyles[1],
             label=r'$r=7,\beta=0.2$')

sns.lineplot(x=alpha_values, y=alpha_values_r_lt_beta_high, ax=ax1,
             color='black', linestyle=linestyles[2],
             label=r'$r=2,\beta=2$')

sns.lineplot(x=alpha_values, y=alpha_values_r_lt_beta_low, ax=ax1,
             color='black', linestyle=linestyles[3],
             label=r'$r=2,\beta=0.2$')

ax1.set_xlabel(r'$\alpha_i$')
ax1.set_ylabel(r'$p_i$')
ax1.legend()

r_values = np.linspace(1, 10, 200)
alpha_grid = np.linspace(0.1, 10, 200)

R_grid, A_grid = np.meshgrid(r_values, alpha_grid, indexing='ij')

pc_vals_beta_zero = get_individual_player_val(
    alpha=A_grid,
    r=R_grid,
    N=N,
    beta=0,
    mu=standard_mu
)

im = ax2.imshow(
    pc_vals_beta_zero,
    extent=[r_values.min(), r_values.max(),
            alpha_grid.min(), alpha_grid.max()],
    origin='lower',
    aspect='auto',
    cmap='Greys_r'
)

fig.colorbar(im, ax=ax2, label=r'$p_i$')

ax2.set_xlabel('r')
ax2.set_ylabel(r'$\alpha_i$')
ax2.set_title(r'Heatmap ($\beta=0$)')

mu_values = np.linspace(0, 0.5, 200)

R, MU = np.meshgrid(r_values, mu_values, indexing='ij')

d_mu_over_r = np.array([
    np.gradient(
        [
            get_individual_player_val(standard_alpha, r, N, 0.5, mu)
            for mu in mu_values
        ],
        mu_values
    )
    for r in r_values
])

ax3.plot_surface(R, MU, d_mu_over_r, cmap=black_grey)
ax3.set_xlabel(r'$r_i$')
ax3.set_ylabel(r'$\mu$')
ax3.set_zlabel(r'$\frac{\partial p_i}{\partial \mu}$')

R, A = np.meshgrid(r_values, alpha_values, indexing='ij')

d_alpha_over_r = np.array([
    np.gradient(
        [
            get_individual_player_val(alpha=alpha, r=r, N=N, beta=0.5, mu=standard_mu)
            for alpha in alpha_values
        ],
        alpha_values
    )
    for r in r_values
])

ax4.plot_surface(R, A, d_alpha_over_r, cmap=black_grey)
ax4.set_xlabel(r'$r_i$')
ax4.set_ylabel(r'$\alpha_i$')
ax4.set_zlabel(r'$\frac{\partial p_i}{\partial \alpha_i}$')

R, B = np.meshgrid(r_values, beta_values, indexing='ij')

d_beta_over_r = np.array([
    np.gradient(
        [
            get_individual_player_val(alpha=standard_alpha, r=r, N=N, beta=beta, mu=standard_mu)
            for beta in beta_values
        ],
        beta_values
    )
    for r in r_values
])

ax5.plot_surface(R, B, d_beta_over_r, cmap=black_grey)
ax5.set_xlabel(r'$r_i$')
ax5.set_ylabel(r'$\beta_i$')
ax5.set_zlabel(r'$\frac{\partial p_i}{\partial \beta_i}$')

for ax in [ax3, ax4, ax5]:
    ax.view_init(elev=25, azim=200)

plt.tight_layout()
plt.savefig(file_path.parent / "main.pdf")