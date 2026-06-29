import numpy as np
import json
from objectives import BaseballModel


# ==============================================================================
# EXECUTION & VERIFICATION TEST SCRIPT
# ==============================================================================
if __name__ == "__main__":
    # 1. Simulate reading your 'player_profiles_LAD_2024.json' file
    # This matches the exact structured format you provided earlier
    json_path = '/home/tlchen/MCMC_circular_data/results/player_profiles_LAD_2024.json'
    with open(json_path, 'r') as f:
        player_profiles = json.load(f)
        
    # 2. Instantiate the Analytical Baseball Model
    print("Initializing BaseballModel engine...")
    model = BaseballModel(player_profiles)

    # 3. Define two different lineups to test absolute generalizability and sorting variance
    # Lineup A
    lineup_a = [8, 7, 6, 5, 4, 3, 2, 1, 0]
    
    # Lineup B: Custom Order
    lineup_b = [2, 1, 4, 0, 3, 7, 5, 8, 6]

    # 4. Calculate exact analytical scores for 9 innings
    runs_lineup_a = model.evaluate_lineup_analytically(lineup_a, innings=9)
    runs_lineup_b = model.evaluate_lineup_analytically(lineup_b, innings=9)

    # 5. Print Results out cleanly to terminal
    print("\n" + "="*55)
    print(" BUKIET (1997) EXACT EVALUATION REPORT")
    print("="*55)
    
    print("Lineup A (Standard Order):")
    for i, idx in enumerate(lineup_a):
        print(f"  {i+1}. {model.name_dict[idx]}")
    print(f"Expected Score per 9-Inning Game: {runs_lineup_a:.4f} runs\n")
    
    print("Lineup B (Custom Order):")
    for i, idx in enumerate(lineup_b):
        print(f"  {i+1}. {model.name_dict[idx]}")
    print(f"Expected Score per 9-Inning Game: {runs_lineup_b:.4f} runs")
    
    print("="*55)
    print(f"Difference between Lineups: {abs(runs_lineup_a - runs_lineup_b):.4f} runs per game.")
    print("="*55)