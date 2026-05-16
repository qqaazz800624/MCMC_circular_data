
import numpy as np
import matplotlib.pyplot as plt
import os
import json  
import argparse

def main():

    parser = argparse.ArgumentParser(description="Analyze MCMC chain histories and plot improvement trajectories")
    parser.add_argument("--save_dir_plots", type=str, default="results/plots", help="Directory to save the output plots")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save the logs")
    parser.add_argument("--experiment_name", type=str, default="tau_5_1000steps", help="Name for this experiment (used in output filenames)")
    parser.add_argument("--team", type=str, default="LAD", help="Team name to analyze (used in input filenames)")
    parser.add_argument("--year", type=str, default="2024", help="Year of the season to analyze (used in input filenames)")
    args = parser.parse_args()

    save_dir_plots = args.save_dir_plots
    save_dir = args.save_dir
    experiment_name = args.experiment_name
    team = args.team
    year = args.year

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

    all_improvement_logs = {}

    print("=" * 60)
    print("Improvement Logs for each proposal:")
    print("=" * 60)

    for prop, color, label in zip(proposals, colors, labels):
        file_path = f"results/mcmc_traces_{experiment_name}_{team}_{year}_{prop}_1chains.npy.npz"
        
        try:
            data = np.load(file_path)
            scores = data['scores']
            chain_0_scores = scores[0]
            
            best_scores_trend = np.maximum.accumulate(chain_0_scores)
            is_new_high = np.concatenate(([True], best_scores_trend[1:] > best_scores_trend[:-1]))
            new_high_steps = np.where(is_new_high)[0]
            new_high_scores = best_scores_trend[is_new_high]
            

            plt.plot(best_scores_trend, linewidth=2.5, color=color, label=label, alpha=0.85)
            
            log_entries = []
            for step, score in zip(new_high_steps, new_high_scores):
                if step == 0:
                    log_entries.append(f"Start ({score:.3f})")
                else:
                    log_entries.append(f"{step} ({score:.3f})")
                    
            trajectory_str = " -> ".join(log_entries)
            
            print(f"{label}:\n{trajectory_str}\n")
            
            all_improvement_logs[prop] = {
                "display_name": label,
                "trajectory_string": trajectory_str,
                "raw_data": {
                    "steps": new_high_steps.tolist(),    
                    "scores": new_high_scores.tolist()
                }
            }
            
        except FileNotFoundError:
            print(f"Cannot find file: {file_path}\n")

    plt.title(f'Comparison of Proposals: {experiment_name} ({team} {year})', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('MCMC Steps', fontsize=14)
    plt.ylabel('Expected Runs (Best Score Found)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
    plt.tight_layout()
    
    output_img = os.path.join(save_dir_plots, f"proposals_comparison_{experiment_name}_{team}_{year}.png")
    plt.savefig(output_img, dpi=300)

    output_json = os.path.join(save_dir, f"proposals_logs_{experiment_name}_{team}_{year}.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_improvement_logs, f, indent=4, ensure_ascii=False)

    print("=" * 60)
    print(f"Comparison plot saved as {output_img}")
    print(f"Improvement logs saved as {output_json}")
    print("=" * 60)

if __name__ == "__main__":
    main()