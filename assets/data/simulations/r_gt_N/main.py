import ludics
import ludics.fitness_functions
import pandas as pd
import numpy as np
import pathlib
import stet

file_path = pathlib.Path(__file__)

def get_pc_from_sims(alphas,r,N, seed):
    initial_state=np.array([0 for _ in range(N)])
    initial_state[1] = 1
    _, dict = ludics.simulate_markov_chain(initial_state=initial_state, number_of_strategies=2, fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function, compute_transition_probability=ludics.compute_introspection_transition_probability, individual_to_action_mutation_probability=np.tile(np.array([0.15, 0.05]), (N, 1)), r=r, contribution_vector=alphas, warmup=100, iterations=100000, seed=seed,
    choice_intensity=0.5)
    per_state_pc = np.array([sum(key) * value / 100000 for key, value in dict.items()])

    sim_pc = np.sum(per_state_pc)/N
    return sim_pc

results_path = file_path.parent / "main.csv"


@stet.once(store=str(file_path.parent / "_stet_store.csv"), key=["N", "seed"])
def record_simulation(N, seed):
    alphas = np.array([i for i in range(N)])
    sim_pc = get_pc_from_sims(alphas=alphas, r=2 * N, N=N, seed=seed)
    pd.DataFrame([[N, seed, sim_pc]], columns=["N", "seed", "p_C"]).to_csv(
        results_path,
        mode="a",
        header=not results_path.exists(),
        index=False,
    )


n_values = list(range(11, 100, 11))
seed_values = list(range(1, 100))
total_runs = len(n_values) * len(seed_values)
completed = 0
for N in n_values:
    for seed in seed_values:
        completed += 1
        print(
            f"\r{file_path.parent.name}: {completed}/{total_runs} (N={N}, seed={seed})",
            end="",
            flush=True,
        )
        record_simulation(N=N, seed=seed)
print()


    
