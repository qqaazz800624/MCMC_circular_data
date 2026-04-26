import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time

import proposals
from utils import run_baseball_mcmc


player_profiles = [
    {'id': 0, 'name': 'Ohtani',  'OUT': 0.612, 'BB': 0.119, '1B': 0.134, '2B': 0.052, '3B': 0.010, 'HR': 0.073},
    {'id': 1, 'name': 'Betts',   'OUT': 0.630, 'BB': 0.118, '1B': 0.159, '2B': 0.046, '3B': 0.010, 'HR': 0.037},
    {'id': 2, 'name': 'Freeman', 'OUT': 0.638, 'BB': 0.122, '1B': 0.147, '2B': 0.055, '3B': 0.003, 'HR': 0.035},
    {'id': 3, 'name': 'Teoscar', 'OUT': 0.673, 'BB': 0.081, '1B': 0.143, '2B': 0.049, '3B': 0.003, 'HR': 0.051},
    {'id': 4, 'name': 'Muncy',   'OUT': 0.648, 'BB': 0.160, '1B': 0.082, '2B': 0.058, '3B': 0.000, 'HR': 0.052},
    {'id': 5, 'name': 'Smith',   'OUT': 0.693, 'BB': 0.094, '1B': 0.131, '2B': 0.044, '3B': 0.002, 'HR': 0.036},
    {'id': 6, 'name': 'Lux',     'OUT': 0.700, 'BB': 0.086, '1B': 0.140, '2B': 0.049, '3B': 0.004, 'HR': 0.021},
    {'id': 7, 'name': 'Edman',   'OUT': 0.686, 'BB': 0.072, '1B': 0.170, '2B': 0.033, '3B': 0.000, 'HR': 0.039},
    {'id': 8, 'name': 'Rojas',   'OUT': 0.697, 'BB': 0.039, '1B': 0.184, '2B': 0.062, '3B': 0.000, 'HR': 0.018}
]

event_types = ['OUT', 'BB', '1B', '2B', '3B', 'HR']
prob_matrix = np.array([[p[event] for event in event_types] for p in player_profiles])
name_dict = {p['id']: p['name'] for p in player_profiles}


def print_lineup(x_indices, score, title):
    print(f"\n[{title}] - expected runs: {score:.3f} runs/game")
    names = [name_dict[idx] for idx in x_indices]
    for i, name in enumerate(names):
        print(f"  Batting order {i+1}: {name}")


def main():
    parser = argparse.ArgumentParser(description="Baseball Lineup MCMC Optimizer")
    parser.add_argument("--max_steps", type=int, default=10000, help="MCMC steps")
    parser.add_argument("--num_sims", type=int, default=1000, help="Number of simulations per step")
    parser.add_argument("--proposal", type=str, default="directional_reversal_proposal", help="Proposal method")
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature parameter (small differences in baseball scores, recommended 0.1~0.5)")
    args = parser.parse_args()

    proposal_func = getattr(proposals, args.proposal)
    
    original_x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    
    print("="*50)
    print("Baseball Lineup MCMC Simulation Optimization Started")
    print("="*50)
    print(f"Settings: {args.max_steps} MCMC steps, {args.num_sims} simulations per step, Proposal: {args.proposal}")
    
    print("\nEvaluating original lineup (baseline)...")
    original_F = objectives.objective_baseball_sim(original_x, prob_matrix, num_simulations=5000) # Baseline evaluation
    print_lineup(original_x, original_F, "Original Dodgers Lineup")
    
    print("\n Starting MCMC to find better lineup...")
    start_time = time.time()
    best_x, best_F = run_baseball_mcmc(original_x, proposal_func, args.num_sims, args.max_steps, args.tau)
    elapsed_time = time.time() - start_time
    
    final_best_F = objectives.objective_baseball_sim(best_x, prob_matrix, num_simulations=5000)
    
    print("\n" + "="*50)
    print(f"Optimization completed! (Elapsed time: {elapsed_time:.1f} seconds)")
    print("="*50)
    print_lineup(best_x, final_best_F, "MCMC Optimized Lineup")
    
    improvement = final_best_F - original_F
    print(f"\nCompared to the original lineup, expected runs per game change: {improvement:+.3f} runs")

if __name__ == "__main__":
    main()