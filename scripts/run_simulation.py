#%%

import numpy as np
from tqdm import tqdm
import argparse

import objectives
import proposals

#%%

def main():
    parser = argparse.ArgumentParser(description="Run MCMC Simulation for Permutation Optimization")
    parser.add_argument("--steps", type=int, default=10000, help="Number of MCMC iterations")
    parser.add_argument("--n", type=int, default=9, help="Number of elements in the permutation")
    parser.add_argument("--objective", type=str, default="toy_objective_1", help="Objective function to optimize")
    parser.add_argument("--proposal", type=str, default="random_swap_proposal", help="Proposal function for generating candidate states")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for the linear term in the objective function")
    parser.add_argument("--beta", type=float, default=0.005, help="Weight for the interaction term in the objective function")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature parameter for the Metropolis-Hastings acceptance criterion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    np.random.seed(args.seed)

    try:
        objective_func = getattr(objectives, args.objective)
        proposal_func = getattr(proposals, args.proposal)
    except AttributeError as e:
        print(f"Error: {e}")
        print("Please ensure that the specified objective and proposal functions exist in their respective modules.")
        return

    current_x = np.arange(1, args.n + 1)
    np.random.shuffle(current_x)
    
    g = np.arange(args.n, 0, -1)

    current_F = objective_func(current_x, g, args.alpha, args.beta)

    best_x = current_x.copy()
    best_F = current_F

    history_F = [current_F]
    acceptance_count = 0

    for t in tqdm(range(args.steps), desc="Running MCMC Simulation"):
        proposed_x = proposal_func(current_x)

        proposed_F = objective_func(proposed_x, g, args.alpha, args.beta)
        delta_F = proposed_F - current_F

        if delta_F > 0:
            acceptance_prob = 1.0
        else:
            acceptance_prob = np.exp(delta_F / args.tau)

        if np.random.rand() <= acceptance_prob:
            current_x = proposed_x
            current_F = proposed_F
            acceptance_count += 1

            if current_F > best_F:
                best_x = current_x.copy()
                best_F = current_F

        history_F.append(current_F)
    acceptance_rate = acceptance_count / args.steps
    return best_x, best_F, history_F, acceptance_rate

#%%

if __name__ == "__main__":
    best_x, best_F, history_F, acceptance_rate = main()
    print("\n--- Simulation Results ---")
    print(f"Best permutation found: {best_x}")
    print(f"Best objective value: {best_F:.4f}")
    print(f"Acceptance rate: {acceptance_rate:.4%}")




#%%