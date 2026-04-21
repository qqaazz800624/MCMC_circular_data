import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_all_ecdfs(data_dir="/home/qqaazz800624/MCMC_circular_data/results",
                   save_dir="results", 
                   tau=5.0):

    proposals = [
        "random_swap_proposal",
        "random_insertion_proposal",
        "directional_reversal_proposal",
        "k_cycle_shift_proposal",
        "block_pair_exchange_proposal"
    ]
    
    plt.figure(figsize=(12, 8))
    colors = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4']

    for i, prop in enumerate(proposals):
        filename = os.path.join(data_dir, f"chain_details_{prop}_tau{tau}.csv")
        
        if not os.path.exists(filename):
            print(f"File not found: {filename}, skipping this method.")
            continue
            
        df = pd.read_csv(filename)
        hitting_times = df['hitting_time'].dropna().values
        
        x = np.sort(hitting_times)
        y = np.arange(1, len(x) + 1) / len(x)
        
        clean_label = prop.replace('_proposal', '').replace('_', ' ').title()
        plt.step(x, y, where='post', linewidth=2.5, color=colors[i % len(colors)], label=clean_label)

    plt.title(f'MCMC Hitting Time Comparison ($\\tau={tau}$)', fontsize=18, fontweight='bold')
    plt.xlabel('Hitting Time (steps)', fontsize=14)
    plt.ylabel('Cumulative Probability (Success Rate)', fontsize=14)
    
    plt.xlim(0, 50000)
    plt.ylim(0, 1.05)
    
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'ecdf_comparison_tau{tau}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_all_ecdfs(save_dir="results", tau=5.0)