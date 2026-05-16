#%%

import numpy as np
import matplotlib.pyplot as plt
import os

save_dir = "results/plots"

proposals = [
    "random_swap_proposal",
    "random_insertion_proposal",
    "directional_reversal_proposal",
    "k_cycle_shift_proposal",
    "block_pair_exchange_proposal"
]

colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231']
labels = [p.replace("_proposal", "").replace("_", " ").title() for p in proposals]
plt.figure(figsize=(14, 7))

for prop, color, label in zip(proposals, colors, labels):

    file_path = f"results/mcmc_traces_tau_5_1000steps_LAD_2024_{prop}_1chains.npy.npz"
    
    try:
        data = np.load(file_path)
        scores = data['scores']
        chain_0_scores = scores[0]
        best_scores_trend = np.maximum.accumulate(chain_0_scores)
        plt.plot(best_scores_trend, linewidth=2.5, color=color, label=label, alpha=0.85)
        
    except FileNotFoundError:
        print(f"Cannot find file: {file_path}")

plt.title('Comparison of MCMC Proposals (LAD 2024)', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('MCMC Steps', fontsize=14)
plt.ylabel('Expected Runs (Best Score Found)', fontsize=14)

plt.grid(True, linestyle='--', alpha=0.6)

plt.legend(fontsize=12, loc='lower right', framealpha=0.9)

plt.tight_layout()
output_img = os.path.join(save_dir, "mcmc_proposals_comparison.png")
plt.savefig(output_img, dpi=300)

print("-" * 50)
print(f"Comparison chart successfully saved as {output_img}")
print("-" * 50)




#%%




#%%






#%%



#%%






#%%