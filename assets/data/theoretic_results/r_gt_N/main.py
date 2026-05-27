import ludics
import ludics.fitness_functions
import pandas as pd
import numpy as np
import pathlib

file_path = pathlib.Path(__file__)

def get_definite_pc(N,r,alphas):
    state_space = ludics.get_state_space(N=N,k=2)

    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
        compute_transition_probability=ludics.compute_introspection_transition_probability,
        individual_to_action_mutation_probability=np.tile(np.array([0.15, 0.05]), (N, 1)),
        contribution_vector=alphas,
        r=r,
        number_of_strategies=2,
        choice_intensity = 0.5
    )

    steady_state = ludics.compute_steady_state(transition_matrix)

    pc = (steady_state @ state_space).sum() / N
    return pc

df = pd.DataFrame(columns = ["N", "p_C"])
df.to_csv(file_path.parent / "main.csv", index=False)

for N in range(3,11):
    alphas = np.array([i for i in range(N)])
    pc = get_definite_pc(alphas=alphas, r=2*N, N=N)
    data = [N, pc]
    df = pd.DataFrame([data], columns=["N", "pc"])
    df.to_csv(file_path.parent / "main.csv",
                  mode="a",
                  header=False,
                  index=False)


    