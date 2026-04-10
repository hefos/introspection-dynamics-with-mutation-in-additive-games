import ludics
import ludics.fitness_functions
import pandas as pd
import numpy as np
import pathlib

file_path = pathlib.Path(__file__)

def get_pc_from_sims(alphas,r,N, seed):
    initial_state=np.array([0 for _ in range(N)])
    initial_state[1] = 1
    _, dict = ludics.simulate_markov_chain(initial_state=initial_state, number_of_strategies=2, fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function, compute_transition_probability=ludics.compute_introspection_transition_probability, individual_to_action_mutation_probability=np.full(shape=(N,2), fill_value=0.1), r=r, contribution_vector=alphas, warmup=100, iterations=10000, seed=seed, 
    choice_intensity=0.5)
    print(dict)
    per_state_pc = np.array([sum(key) * value / 10000 for key, value in dict.items()])

    sim_pc = np.sum(per_state_pc)/N
    return sim_pc

df = pd.Dataframe(columns = ["N", "seed", "p_C"])
df.to_csv(file_path.parent / "main.csv", index=False)

for N in range(11, 100, 11):
    alphas = np.array([i for i in range(N)])
    for seed in range(1,40):
        pc = get_pc_from_sims(alphas=alphas, r=2*N, seed=seed)
        data = [N, seed, pc]
        df = pd.DataFrame(data)
        df.to_csv(file_path.parent / "main.csv",
                  mode="a",
                  header=False,
                  index=False)


    