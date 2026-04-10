import numpy as np
import matplotlib.pyplot as plt
import os

def plot_cdfs(sorted_scores, probabilities, tau, 
              save_dir="results",
              save_path="distribution_cdfs.png"):
    N = len(sorted_scores)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ascending_scores = sorted_scores[::-1] 
    cdf_raw = np.arange(1, N + 1) / N
    ax1.plot(ascending_scores, cdf_raw, color='royalblue', linewidth=2)
    ax1.set_title('CDF of Original Objective Function $F(x)$')
    ax1.set_xlabel('F(x) Score')
    ax1.set_ylabel('Cumulative Probability')
    ax1.grid(True, alpha=0.3)

    top_k = min(1000, N)
    cdf_boltzmann = np.cumsum(probabilities[:top_k])
    ax2.plot(range(1, top_k + 1), cdf_boltzmann, color='crimson', linewidth=2)
    ax2.set_title(f'CDF of Boltzmann Prob. (Top {top_k}, $\\tau={tau}$)')
    ax2.set_xlabel('Rank (1 to 1000, sorted by score)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, save_path), dpi=300)
    plt.show()