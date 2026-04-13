import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from mpl_toolkits.mplot3d import Axes3D

file_path = pathlib.Path(__file__)

def get_individual_player_val(alpha, r, N, beta, mu):
    numerator = 1 - (2 * mu)
    denominator = 1 + np.exp(beta * (alpha * (1 - (r / N))))
    return (numerator / denominator) + mu


fig, axarray = plt.subplots(2, 2, figsize=(10, 8))
ax = axarray.flatten()

N = 5
standard_alpha = 2
standard_mu = 0.1

linestyles = ['-', '--', '-.', ':']

beta_values = np.linspace(0, 2, 200)

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

sns.lineplot(x=beta_values, y=beta_values_r_lt, ax=ax[0],
             color='black', linestyle=linestyles[0], linewidth=2, label=r'$r = 2$')

sns.lineplot(x=beta_values, y=beta_values_r_eq, ax=ax[0],
             color='black', linestyle=linestyles[1], linewidth=2, label=r'$r = 5$')

sns.lineplot(x=beta_values, y=beta_values_r_gt, ax=ax[0],
             color='black', linestyle=linestyles[2], linewidth=2, label=r'$r = 7$')

ax[0].set_xlabel(r'$\beta$')
ax[0].set_ylabel(r'$p_i$')
ax[0].legend()

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

sns.lineplot(x=alpha_values, y=alpha_values_r_gt_beta_high, ax=ax[1],
             color='black', linestyle=linestyles[0], linewidth=2,
             label=r'$r = 7,\ \beta = 2$')

sns.lineplot(x=alpha_values, y=alpha_values_r_gt_beta_low, ax=ax[1],
             color='black', linestyle=linestyles[1], linewidth=2,
             label=r'$r = 7,\ \beta = 0.2$')

sns.lineplot(x=alpha_values, y=alpha_values_r_lt_beta_high, ax=ax[1],
             color='black', linestyle=linestyles[2], linewidth=2,
             label=r'$r = 2,\ \beta = 2$')

sns.lineplot(x=alpha_values, y=alpha_values_r_lt_beta_low, ax=ax[1],
             color='black', linestyle=linestyles[3], linewidth=2,
             label=r'$r = 2,\ \beta = 0.2$')

ax[1].set_xlabel(r'$\alpha_i$')
ax[1].set_ylabel(r'$p_i$')
ax[1].legend()

r_values = np.linspace(1,10,2000)
R,A = np.meshgrid(r_values, alpha_values)
pc_vals_beta_zero = get_individual_player_val(
    alpha=A,
    r=R,
    N=N,
    beta=0,
    mu=standard_mu
)
im = ax[2].imshow(
    pc_vals_beta_zero,
    extent=[r_values.min(), r_values.max(),
            alpha_values.min(), alpha_values.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)

plt.colorbar(im, label=r'$p_i$')

ax[2].set_xlabel('r')
ax[2].set_ylabel(r'$\alpha_i$')
ax[2].set_title(r'Heatmap of $p_i$ ($\beta = 0$)')

mu_values = np.linspace(0, 0.5, 2000, endpoint=False)

d_beta_over_r = np.array([
    np.gradient(np.array([get_individual_player_val(alpha=standard_alpha, N=N, r=r, beta=beta, mu=standard_mu) for beta in beta_values])) for r in r_values
])

R, B = np.meshgrid(r_values, beta_values, indexing='ij')
plot = ax[3].plot_surface(
    R, B, d_beta_over_r,
    cmap='grey', label = r'$\frac{\partial \beta_i}{\partial r_i}$'
)




d_alpha_over_r = np.array([
    np.gradient(np.array([get_individual_player_val(alpha=alpha, N=N, r=r, beta=0.5, mu=standard_mu) for alpha in alpha_values])) for r in r_values
])

d_mu_over_r = np.array([
    np.gradient(np.array([get_individual_player_val(alpha=standard_alpha, N=N, r=r, beta=0.5, mu=mu) for mu in mu_values])) for r in r_values
])


R, A = np.meshgrid(r_values, alpha_values, indexing='ij')
surf = ax[3].plot_surface(
    R, A, d_alpha_over_r,
    cmap='grey', label = r'$\frac{\partial \alpha_i}{\partial r_i}$'
)

R, MU = np.meshgrid(r_values, mu_values, indexing='ij')
surf = ax[3].plot_surface(
    R, MU, d_mu_over_r,
    cmap='grey', label = r'$\frac{\partial \mu_i}{\partial r_i}$'
)

ax[3].set_xlabel(r'$r_i$')
ax[3].set_ylabel(r'$\mu$')
ax[3].set_zlabel(r'$\frac{\partial p_i}{\partial \mu}$')

plt.tight_layout()
plt.savefig(file_path.parent / "main.pdf")