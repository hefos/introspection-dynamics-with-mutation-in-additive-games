import ludics
import ludics.fitness_functions
import pandas as pd
import numpy as np
import pathlib

file_path = pathlib.Path(__file__)

def get_new_formula_val(alphas, r, N):
    
    running_phis = []

    for i in range(N):
        numerator = 0.8
        denominator = 1 + np.exp(0.5 * (alphas[i] * (1 - (r/N))))

        running_phis.append((numerator / denominator) + 0.05)
    
    return sum(running_phis) / N

df = pd.DataFrame(columns = ["N", "p_C"])
df.to_csv(file_path.parent / "main.csv", index=False)

for N in range(3,11):
    alphas = np.array([i for i in range(N)])
    pc = get_new_formula_val(alphas=alphas, r=N/2, N=N)
    data = [N, pc]
    df = pd.DataFrame([data], columns=["N", "pc"])
    df.to_csv(file_path.parent / "main.csv",
                  mode="a",
                  header=False,
                  index=False)

for N in range(11,100):
    alphas = np.array([i for i in range(N)])
    pc = get_new_formula_val(alphas=alphas, r=N/2, N=N)
    data = [N, pc]
    df = pd.DataFrame([data], columns=["N", "pc"])
    df.to_csv(file_path.parent / "main.csv",
                  mode="a",
                  header=False,
                  index=False)
