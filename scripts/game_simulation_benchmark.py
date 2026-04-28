import time
import numpy as np
from objectives import BaseballSimulator


player_profiles = [
    {'id': 0, 'name': 'Ohtani', 'OUT': 0.607, 'BB': 0.123, '1B': 0.134, '2B': 0.052, '3B': 0.01, 'HR': 0.074},
    {'id': 1, 'name': 'Betts', 'OUT': 0.627, 'BB': 0.12, '1B': 0.159, '2B': 0.047, '3B': 0.01, 'HR': 0.037},
    {'id': 2, 'name': 'Freeman', 'OUT': 0.623, 'BB': 0.138, '1B': 0.147, '2B': 0.055, '3B': 0.003, 'HR': 0.034},
    {'id': 3, 'name': 'Teoscar', 'OUT': 0.659, 'BB': 0.095, '1B': 0.143, '2B': 0.049, '3B': 0.003, 'HR': 0.051},
    {'id': 4, 'name': 'Muncy', 'OUT': 0.642, 'BB': 0.171, '1B': 0.078, '2B': 0.058, '3B': 0.0, 'HR': 0.051},
    {'id': 5, 'name': 'Smith', 'OUT': 0.673, 'BB': 0.11, '1B': 0.132, '2B': 0.044, '3B': 0.004, 'HR': 0.037},
    {'id': 6, 'name': 'Lux', 'OUT': 0.68, 'BB': 0.094, '1B': 0.152, '2B': 0.049, '3B': 0.004, 'HR': 0.021},
    {'id': 7, 'name': 'Edman', 'OUT': 0.706, 'BB': 0.078, '1B': 0.137, '2B': 0.033, '3B': 0.007, 'HR': 0.039},
    {'id': 8, 'name': 'Rojas', 'OUT': 0.665, 'BB': 0.077, '1B': 0.178, '2B': 0.062, '3B': 0.0, 'HR': 0.018},
]

def main():
    print("Initiate (Benchmark)...")
    simulator = BaseballSimulator(player_profiles)
    
    test_lineup = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    num_sims = 10000
    
    # Start timing
    start_time = time.time()
    
    score = simulator.evaluate_lineup(test_lineup, num_simulations=num_sims)
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("-" * 40)
    print(f"Simulation : {num_sims} games")
    print(f"Total time: {elapsed_time:.4f} seconds")
    print(f"Expected score: {score:.3f} runs/game")
    print(f"Estimated 1000 MCMC steps time: {(elapsed_time * 1000) / 60:.2f} minutes")
    print("-" * 40)

if __name__ == "__main__":
    main()