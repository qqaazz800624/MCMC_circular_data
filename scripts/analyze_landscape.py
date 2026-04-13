#%%

import numpy as np
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

from objectives import toy_objective_1
from utils import plot_cdfs

def analyze_landscape(n=9, alpha=1.0, beta=0.005, tau=5.0,
                      save_dir="results"):
    
    print(f"--- Initiating Landscape Analysis ---")
    print(f"Parameters: n={n}, alpha={alpha}, beta={beta}, tau={tau}")
    

    print(f"\n1. Generating {n}! permutations...")
    permutations = list(itertools.permutations(range(1, n + 1)))
    N = len(permutations)
    print(f"Total permutations generated: {N}.")

    g = np.arange(n, 0, -1)
    scores = np.zeros(N)
    
    for i in tqdm(range(N), desc="Calculating F(x) scores"):
        scores[i] = toy_objective_1(permutations[i], g=g, alpha=alpha, beta=beta)

    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_perms = [permutations[i] for i in sorted_indices]

    shifted_scores = sorted_scores - np.max(sorted_scores)
    exp_term = np.exp(shifted_scores / tau)
    Z = np.sum(exp_term)
    probabilities = exp_term / Z

    print("\n--- Top State Probability Analysis ---")
    print(f"Top 1 Score: {sorted_scores[0]:.4f} | Boltzmann Prob: {probabilities[0]:.4%}")
    print(f"Top 2 Score: {sorted_scores[1]:.4f} | Boltzmann Prob: {probabilities[1]:.4%}")
    print(f"Top 3 Score: {sorted_scores[2]:.4f} | Boltzmann Prob: {probabilities[2]:.4%}")
    
    if probabilities[0] - probabilities[1] < 0.0001:
        print("WARNING: Top 1 and Top 2 probabilities are too close! Consider DECREASING tau (to concentrate probability on top scores).")
    elif probabilities[0] > 0.5:
        print("WARNING: Top 1 probability exceeds 50%! Consider INCREASING tau (to prevent MCMC from getting stuck in local optima).")
    else:
        print("STATUS OK: Reasonable distinction between the highest and second-highest probabilities.")

    print("\n--- Top 10 Preview (Full Top 100 saved to CSV) ---")
    top_100_data = []
    for i in range(100):
        top_100_data.append({
            "Rank": i + 1,
            "Permutation": str(sorted_perms[i]),
            "Score_F(x)": sorted_scores[i],
            "Probability": probabilities[i]
        })
        if i < 10: 
            print(f"Rank {i+1:2d}: {sorted_perms[i]} | F(x) = {sorted_scores[i]:.4f}")
            
    df_top100 = pd.DataFrame(top_100_data)
    csv_save_path = os.path.join(save_dir, "top_100_permutations.csv")
    df_top100.to_csv(csv_save_path, index=False)

    plot_cdfs(sorted_scores, probabilities, tau, save_dir=save_dir,
              save_path="distribution_cdfs.png")

if __name__ == "__main__":
    analyze_landscape(n=9, alpha=1.0, beta=0.005, tau=2.0, save_dir="results")
    print("\n--- Landscape Analysis Completed ---")

#%%


