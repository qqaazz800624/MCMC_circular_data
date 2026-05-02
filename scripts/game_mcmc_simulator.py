import numpy as np
import time
import proposals
import argparse
import json
import os

from optimizer import MCMCOptimizer
from tqdm import tqdm
from objectives import BaseballSimulator
from utils import generate_initial_states

def main():
    parser = argparse.ArgumentParser(description="Run MCMC Baseball Lineup Optimization")
    parser.add_argument("--num_initials", type=int, default=1, help="Number of random initial lineups to test")
    parser.add_argument("--num_sims_per_step", type=int, default=1000, help="Number of games to evaluate each lineup")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum MCMC steps to run")
    parser.add_argument("--proposal", type=str, default="random_swap_proposal", help="Proposal function to use")
    parser.add_argument("--tau", type=float, default=5, help="Temperature parameter for MCMC")
    parser.add_argument("--data_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--initial_x", type=str, default=None, help="Comma-separated initial lineup (e.g., '0,1,2,3,4,5,6,7,8')")
    parser.add_argument("--lineup_filename", type=str, default="player_profiles_LAD_2024.json", help="Filename for player profiles JSON")

    args = parser.parse_args()

    base_lineup = np.arange(9)
    num_states = args.num_initials

    if args.initial_x:
        parsed_state = np.array(list(map(int, args.initial_x.split(','))))
        
        if len(parsed_state) != 9:
            raise ValueError(f"Expecting 9 elements, but got {len(parsed_state)}.")
            
        initial_states = [parsed_state.copy() for _ in range(num_states)]
        print(f"Using user-provided initial state: {parsed_state} for all {num_states} chains.")
    else:
        initial_states = generate_initial_states(elements=base_lineup, num_states=num_states, seed=42)

    proposal_func = getattr(proposals, args.proposal)

    lineup_path = os.path.join(args.data_dir, args.lineup_filename)
    print(f"Loading player profiles from {lineup_path}...")

    with open(lineup_path, "r") as f:
        player_profiles = json.load(f)

    team_name = args.lineup_filename.replace("player_profiles_", "").replace(".json", "")
    global_cache_path = os.path.join(args.data_dir, f"global_cache_{team_name}.json")

    global_cache = {}
    if os.path.exists(global_cache_path):
        print(f"Found existing global cache at {global_cache_path}. Loading...")
        with open(global_cache_path, "r", encoding="utf-8") as f:
            raw_cache = json.load(f)
            for k, v in raw_cache.items():
                tuple_key = tuple(map(int, k.split(',')))
                global_cache[tuple_key] = v
        print(f"Loaded {len(global_cache)} previously simulated lineups!")
    else:
        print("No existing global cache found. Starting fresh.")

    initial_cache_size = len(global_cache)

    simulator = BaseballSimulator(player_profiles)
    optimizer = MCMCOptimizer(simulator, 
                              num_sims_per_step=args.num_sims_per_step, 
                              tau=args.tau,
                              initial_cache=global_cache)

    print("=" * 50)
    print(f"MCMC Simulation Starting ({args.num_initials} chains)...")
    print("=" * 50)

    overall_best_score = -float('inf')
    overall_best_lineup = None
    overall_best_chain_idx = -1
    overall_best_step = -1
    total_steps_explored_all_chains = 0

    all_visited_states_this_run = set()

    start_time = time.time()
    
    #for i in tqdm(range(args.num_initials), desc="Running MCMC Chains"):
    for i in range(args.num_initials):
        result = optimizer.optimize(
            initial_x=initial_states[i],
            proposal_func=proposal_func,  
            max_steps=args.max_steps         
        )
        
        total_steps_explored_all_chains += result['total_steps']
        all_visited_states_this_run.update(result['visited_states'])
        
        if result['best_score'] > overall_best_score:
            overall_best_score = result['best_score']
            overall_best_lineup = result['best_lineup']
            overall_best_chain_idx = i
            overall_best_step = result['steps_to_best']
    
    end_time = time.time()
    execution_time = end_time - start_time

    # statistics to record
    final_cache_size = len(optimizer.score_cache)
    new_simulations_this_run = final_cache_size - initial_cache_size
    unique_combinations_explored = len(all_visited_states_this_run)

    total_evaluations = (args.num_initials * 1) + total_steps_explored_all_chains
    cache_hits = total_evaluations - new_simulations_this_run
    cache_ratio = cache_hits / total_evaluations if total_evaluations > 0 else 0
    best_names = [simulator.name_dict[idx] for idx in overall_best_lineup]

    # print results
    print(f"\nTest Successfully Completed!")
    print(f"Total time: {execution_time:.2f} seconds")
    print("-" * 50)
    print("Cache Hits and Performance Data:")
    print(f"   - Total steps evaluated this run: {total_evaluations}")
    print(f"   - Unique combinations explored (The 'Exploration Power'): {unique_combinations_explored}")
    print(f"   - New lineups actually simulated (Simulator called): {new_simulations_this_run}")
    print(f"   - Cache hits (avoided simulations): {cache_hits}")
    print(f"   - Effective Cache hit ratio: {cache_ratio:.1%}")
    print(f"   - Total Global Cache size now: {final_cache_size} / 362880")
    print("-" * 50)
    print(f"Best lineup found overall: {overall_best_lineup}")
    print(f"Best expected score: {overall_best_score:.3f}")
    print(f"Found in Chain #{overall_best_chain_idx} at step {overall_best_step}")
    print("=" * 50)

    print(f"Saving updated global cache to {global_cache_path}...")
    save_cache = {','.join(map(str, k)): v for k, v in optimizer.score_cache.items()}
    with open(global_cache_path, "w", encoding="utf-8") as f:
        json.dump(save_cache, f, ensure_ascii=False)

    print("Saving results data...")
    results_data = {
        "experiment_config": {
            "lineup_filename": args.lineup_filename,
            "num_initials": args.num_initials,
            "num_sims_per_step": args.num_sims_per_step,
            "max_steps": args.max_steps,
            "proposal": args.proposal,
            "tau": args.tau
        },
        "performance": {
            "execution_time_seconds": round(execution_time, 2),
            "total_evaluations": total_evaluations,
            "unique_combinations_explored": unique_combinations_explored, 
            "new_simulations_this_run": new_simulations_this_run,
            "total_global_cache_size": final_cache_size,
            "cache_hits": cache_hits,
            "cache_hit_ratio": round(cache_ratio, 4)
        },
        "best_result": {
            "best_lineup_ids": [int(idx) for idx in overall_best_lineup],
            "best_lineup_names": best_names,
            "best_expected_score": round(overall_best_score, 3),
            "found_in_chain": overall_best_chain_idx,
            "found_at_step": overall_best_step
        }
    }

    output_filename = f"mcmc_results_{team_name}_{args.proposal}_{args.num_initials}chains.json"
    output_path = os.path.join(args.data_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"Results successfully saved to {output_path}")

if __name__ == "__main__":
    main()