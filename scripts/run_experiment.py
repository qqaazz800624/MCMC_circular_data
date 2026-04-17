#%%

import numpy as np
from tqdm import tqdm
import argparse
import objectives
import proposals
from utils import generate_initial_states, run_single_mcmc_chain

#%%

def main():
    parser = argparse.ArgumentParser(description="Run 10000-Initial MCMC Experiment")
    parser.add_argument("--num_experiments", type=int, default=10000, help="Number of random initials")
    parser.add_argument("--max_steps", type=int, default=50000, help="Max steps before giving up")
    parser.add_argument("--n", type=int, default=9, help="Number of elements")
    parser.add_argument("--objective", type=str, default="toy_objective_1")
    parser.add_argument("--proposal", type=str, default="random_swap_proposal")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--true_max_F", type=float, required=True, help="The known global maximum F(x)")
    
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    objective_func = getattr(objectives, args.objective)
    proposal_func = getattr(proposals, args.proposal)

    print(f"Generating {args.num_experiments} fair initial states...")
    initial_states = generate_initial_states(num_states=args.num_experiments, n=args.n, seed=args.seed)
    g = np.linspace(4.9, 4.1, num=args.n) 

    hitting_times = []
    failed_count = 0

    print(f"\n--- Starting Experiment: {args.proposal} ---")
    for i in tqdm(range(args.num_experiments), desc="Simulating Chains"):
        h_time = run_single_mcmc_chain(
            initial_states[i], objective_func, proposal_func, g, 
            args.alpha, args.beta, args.tau, args.true_max_F, args.max_steps
        )
        
        hitting_times.append(h_time)
        if h_time == args.max_steps:
            failed_count += 1

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

if __name__ == "__main__":
    main()



#%%






#%%







#%%





#%%






#%%