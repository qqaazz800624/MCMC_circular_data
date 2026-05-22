import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    os.makedirs(save_dir_plots, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

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
        file_path = os.path.join(save_dir, f"mcmc_traces_{experiment_name}_{team}_{year}_{prop}_1chains.npy.npz")
        
        try:
            data = np.load(file_path)
            chain_scores = data['scores'][0]
            chain_states = data['states'][0]
            
            best_scores_trend = np.maximum.accumulate(chain_scores)
            plt.plot(best_scores_trend, linewidth=2.5, color=color, label=label, alpha=0.85, zorder=2)
            
            max_score_so_far = -float('inf')
            reference_best_state = None
            log_entries = []
            raw_log_data = []

            for step in range(len(chain_scores)):
                current_score = chain_scores[step]
                current_state = chain_states[step]

                if step == 0:
                    max_score_so_far = current_score
                    reference_best_state = current_state.copy()
                    log_entries.append(f"Start ({max_score_so_far:.3f})")
                    raw_log_data.append({"step": step, "score": float(current_score), "state": current_state.tolist(), "note": "Start"})
                    continue

                if current_score > max_score_so_far + 1e-6:
                    max_score_so_far = current_score
                    reference_best_state = current_state.copy()
                    log_entries.append(f"{step} ({max_score_so_far:.3f})")
                    raw_log_data.append({"step": step, "score": float(current_score), "state": current_state.tolist(), "note": "New High"})
                
                elif abs(current_score - max_score_so_far) <= 1e-6:
                    if np.array_equal(current_state, reference_best_state):
                        log_entries.append(f"{step} (Revisit)")
                        raw_log_data.append({"step": step, "score": float(current_score), "state": current_state.tolist(), "note": "Revisit Best"})
                    else:
                        reference_best_state = current_state.copy()
                        log_entries.append(f"{step} (Tie)")
                        raw_log_data.append({"step": step, "score": float(current_score), "state": current_state.tolist(), "note": "Tie"})
            
            revisit_steps = [item['step'] for item in raw_log_data if item['note'] == "Revisit Best"]
            revisit_scores = [item['score'] for item in raw_log_data if item['note'] == "Revisit Best"]
            
            if revisit_steps:
                plt.scatter(revisit_steps, revisit_scores, color=color, marker='x', s=40, alpha=0.7, zorder=3)
                    
            trajectory_str = " -> ".join(log_entries)
            print(f"{label}:\n{trajectory_str}\n")
            
            all_improvement_logs[prop] = {
                "display_name": label,
                "trajectory_string": trajectory_str,
                "raw_data": raw_log_data
            }
            
        except FileNotFoundError:
            print(f"Cannot find file: {file_path}\n")

    plt.title(f'Comparison of Proposals: {experiment_name} ({team} {year})', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('MCMC Steps', fontsize=14)
    plt.ylabel('Expected Runs (Best Score Found)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    handles, lgd_labels = plt.gca().get_legend_handles_labels()
    revisit_marker = Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=8, label='Revisit Best')
    handles.append(revisit_marker)
    lgd_labels.append('Revisit Best (Stuck)')
    
    plt.legend(handles=handles, labels=lgd_labels, fontsize=12, loc='lower right', framealpha=0.9)
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

