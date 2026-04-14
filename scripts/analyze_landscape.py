#%%

import numpy as np
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import argparse
from objectives import toy_objective_1
from utils import plot_cdfs

def main():
    parser = argparse.ArgumentParser(description="Analyze the landscape of the objective function over all permutations")   
    parser.add_argument("--n", type=int, default=9, help="Number of elements in the permutation")
    parser.add_argument("--g", type=float, nargs='+', default=[4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1], help="Values for g (should match n)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for the linear term in the objective function")
    parser.add_argument("--beta", type=float, default=10, help="Weight for the interaction term in the objective function")
    parser.add_argument("--tau", type=float, default=5.0, help="Temperature parameter for Boltzmann distribution")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results and plots")

    args = parser.parse_args()
    n = args.n
    g = args.g
    alpha = args.alpha
    beta = args.beta
    tau = args.tau
    save_dir = args.save_dir

    print(f"--- Initiating Landscape Analysis ---")
    print(f"Parameters: n={n}, g={g}, alpha={alpha}, beta={beta}, tau={tau}")
    

    print(f"\n1. Generating {n}! permutations...")
    elements = np.linspace(0.8, 0.6, num=n).tolist()
    permutations = list(itertools.permutations(elements))
    N = len(permutations)
    print(f"Total permutations generated: {N}.")

    scores = np.zeros(N)
    linear_terms = np.zeros(N)
    product_terms = np.zeros(N)
    
    for i in tqdm(range(N), desc="Calculating F(x) scores"):
        scores[i], linear_terms[i], product_terms[i] = toy_objective_1(permutations[i], g=g, alpha=alpha, beta=beta)

    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_perms = [permutations[i] for i in sorted_indices]


    weighted_linear = alpha * linear_terms
    weighted_product = beta * product_terms

    print(f"\n--- Objective Function Decomposition (Weighted) ---")
    print(f"Linear Term (alpha={alpha}): Mean = {np.mean(weighted_linear):.4f}, Std = {np.std(weighted_linear):.4f}")
    print(f"Product Term (beta={beta}): Mean = {np.mean(weighted_product):.4f}, Std = {np.std(weighted_product):.4f}")

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
            "Linear_Term": linear_terms[sorted_indices[i]],
            "Product_Term": product_terms[sorted_indices[i]],
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
    main()
    print("\n--- Landscape Analysis Completed ---")

#%%


