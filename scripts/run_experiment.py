#%%

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import objectives
import proposals
from utils import generate_initial_states, run_single_mcmc_chain
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Run 10000-Initial MCMC Experiment")
    parser.add_argument("--num_experiments", type=int, default=10000, help="Number of random initials")
    parser.add_argument("--max_steps", type=int, default=50000, help="Max steps before giving up")
    parser.add_argument("--n", type=int, default=9, help="Number of elements")
    parser.add_argument("--g", type=float, nargs='+', default=[4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1], help="Values for g (should match n)")
    parser.add_argument("--x", type=float, nargs='+', default=[0.8, 0.775, 0.75, 0.725, 0.7, 0.675, 0.65, 0.625, 0.6], help="Elements to be permuted (should match n)")
    parser.add_argument("--objective", type=str, default="toy_objective_1")
    parser.add_argument("--proposal", type=str, default="random_swap_proposal")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=5)
    parser.add_argument("--true_max_F", type=float, required=True, help="The known global maximum F(x)")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
    
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    objective_func = getattr(objectives, args.objective)
    proposal_func = getattr(proposals, args.proposal)

    print(f"Generating {args.num_experiments} fair initial states...")
    initial_states = generate_initial_states(elements=args.x, num_states=args.num_experiments, seed=args.seed)
    g = np.array(args.g)

    hitting_times = []
    chain_details = []
    failed_count = 0

    print(f"\n--- Starting Experiment: {args.proposal} ---")
    for i in tqdm(range(args.num_experiments), desc="Simulating Chains"):
        final_x, final_F, h_time = run_single_mcmc_chain(
            initial_states[i], objective_func, proposal_func, g, 
            args.alpha, args.beta, args.tau, args.true_max_F, args.max_steps
        )
        
        hitting_times.append(h_time)
        if h_time == args.max_steps:
            failed_count += 1

        chain_details.append({
            "chain_id": i + 1,
            "hitting_time": h_time,
            "final_F": final_F,
            "final_x": str(final_x.tolist()), # 轉成乾淨的 string 格式
            "success": h_time < args.max_steps
        })

    hitting_times = np.array(hitting_times)
    mean_hitting_time = np.mean(hitting_times)
    std_hitting_time = np.std(hitting_times)
    success_rate = 1.0 - (failed_count / args.num_experiments)

    print("\n" + "="*40)
    print("EXPERIMENT RESULTS")
    print("="*40)
    print(f"Method:        {args.proposal}")
    print(f"Temperature:   tau = {args.tau}")
    print(f"Total Chains:  {args.num_experiments}")
    print(f"Success Rate:  {success_rate:.2%} (Chains that hit global max before {args.max_steps} steps)")
    print(f"Hitting Time:  Mean = {mean_hitting_time:.2f} steps, Std = {std_hitting_time:.2f}")
    print("="*40)

    print("\nSaving results...")

    summary_results = {
        "method": args.proposal,
        "tau": args.tau,
        "alpha": args.alpha,
        "beta": args.beta,
        "total_chains": args.num_experiments,
        "max_steps": args.max_steps,
        "success_rate": success_rate,
        "mean_hitting_time": mean_hitting_time,
        "std_hitting_time": std_hitting_time,
        "failed_count": failed_count
    }

    summary_filename = os.path.join(args.save_dir, f"summary_{args.proposal}_tau{args.tau}.json")
    with open(summary_filename, 'w') as f:
        json.dump(summary_results, f, indent=4)

    df_details = pd.DataFrame(chain_details)
    csv_filename = os.path.join(args.save_dir, f"chain_details_{args.proposal}_tau{args.tau}.csv")
    df_details.to_csv(csv_filename, index=False)

    print(f"✅ Results successfully saved to the '{args.save_dir}' directory.")

if __name__ == "__main__":
    main()



#%%






#%%







#%%





#%%






#%%