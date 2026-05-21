# An Exact Cooperation Formula for Introspection Dynamics in the Heterogeneous Public Goods Game

This repository contains the source code and data for the paper of the same
name. We prove that introspection dynamics on any state-independent game is
exactly solvable, and derive a closed-form cooperation probability for the
heterogeneous public goods game.

## Regenerating the figures

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment.
With `uv` installed, run the following from the repository root.

**Figure 1** (validation against exact values and simulation):

```bash
uv run python assets/graphs/simulations/main.py
```

**Figure 2** (structural properties of \(p_i\)):

```bash
uv run python assets/graphs/big_panel/main.py
```

Both scripts write their output directly to `tex/`. The simulation figure
script also regenerates `tex/stationary_table.tex`.

## Regenerating the underlying data

Figure 1 reads from CSV files under `assets/data/`, which are committed to
the repository. Each of the two cases (`r_gt_N` and `r_lt_N`) has three
data sources: Monte Carlo simulations, exact theoretic results, and the new
formula. To recompute them (note: the simulation runs are long):

```bash
uv run python assets/data/simulations/r_gt_N/main.py
uv run python assets/data/simulations/r_lt_N/main.py
uv run python assets/data/theoretic_results/r_gt_N/main.py
uv run python assets/data/theoretic_results/r_lt_N/main.py
uv run python assets/data/new_formula/r_gt_N/main.py
uv run python assets/data/new_formula/r_lt_N/main.py
```

Figure 2 is computed analytically and requires no data scripts.
