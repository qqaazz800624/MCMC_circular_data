import numpy as np
import json
from objectives import BaseballModel


def main():
    json_path = '/home/qqaazz800624/MCMC_circular_data/results/player_profiles_SF_2024.json'
    with open(json_path, 'r') as f:
        player_profiles = json.load(f)
        
    print("Initializing BaseballModel engine...")
    model = BaseballModel(player_profiles)

    lineup = [0, 1, 2, 3, 4, 5, 6, 7, 8]  

    runs_lineup = model.evaluate_lineup_analytically(lineup, innings=9)

    print("\n" + "="*55)
    print(" BUKIET (1997) EXACT EVALUATION REPORT")
    print("="*55)
    
    print("Lineup (Custom Order):")
    for i, idx in enumerate(lineup):
        print(f"  {i+1}. {model.name_dict[idx]}")
    print(f"Expected Score per 9-Inning Game: {runs_lineup:.4f} runs\n")

if __name__ == "__main__":
    main()