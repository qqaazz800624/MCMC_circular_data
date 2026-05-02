import time
import numpy as np
from objectives import BaseballSimulator

# LAD (2024) R/G: 5.20 Simulated R/G: 5.36 / 5.374
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

# NYY (2024) R/G: 5.03 Simulated R/G: 4.93 / 4.909
# player_profiles = [
#     {'id': 0, 'name': 'Torres', 'OUT': 0.671, 'BB': 0.102, '1B': 0.165, '2B': 0.039, '3B': 0.0, 'HR': 0.023},
#     {'id': 1, 'name': 'Soto', 'OUT': 0.58, 'BB': 0.187, '1B': 0.126, '2B': 0.043, '3B': 0.006, 'HR': 0.058},
#     {'id': 2, 'name': 'Judge', 'OUT': 0.542, 'BB': 0.203, '1B': 0.121, '2B': 0.051, '3B': 0.001, 'HR': 0.082},
#     {'id': 3, 'name': 'Stanton', 'OUT': 0.701, 'BB': 0.087, '1B': 0.109, '2B': 0.044, '3B': 0.0, 'HR': 0.059},
#     {'id': 4, 'name': 'Chisholm', 'OUT': 0.676, 'BB': 0.092, '1B': 0.153, '2B': 0.034, '3B': 0.006, 'HR': 0.039},
#     {'id': 5, 'name': 'Rizzo', 'OUT': 0.699, 'BB': 0.096, '1B': 0.152, '2B': 0.032, '3B': 0.0, 'HR': 0.021},
#     {'id': 6, 'name': 'Volpe', 'OUT': 0.708, 'BB': 0.068, '1B': 0.158, '2B': 0.039, '3B': 0.01, 'HR': 0.017},
#     {'id': 7, 'name': 'Wells', 'OUT': 0.68, 'BB': 0.126, '1B': 0.118, '2B': 0.043, '3B': 0.002, 'HR': 0.031},
#     {'id': 8, 'name': 'Verdugo', 'OUT': 0.709, 'BB': 0.081, '1B': 0.142, '2B': 0.045, '3B': 0.002, 'HR': 0.021},
# ]

# BOS (2024) R/G: 4.64 Simulated R/G: 4.72 / 4.63
# player_profiles = [
#     {'id': 0, 'name': 'Duran', 'OUT': 0.658, 'BB': 0.082, '1B': 0.147, '2B': 0.065, '3B': 0.019, 'HR': 0.029},
#     {'id': 1, 'name': 'Devers', 'OUT': 0.646, 'BB': 0.116, '1B': 0.126, '2B': 0.057, '3B': 0.008, 'HR': 0.047},
#     {'id': 2, 'name': "O'Neill", 'OUT': 0.663, 'BB': 0.127, '1B': 0.106, '2B': 0.038, '3B': 0.0, 'HR': 0.066},
#     {'id': 3, 'name': 'Casas', 'OUT': 0.663, 'BB': 0.128, '1B': 0.123, '2B': 0.033, '3B': 0.0, 'HR': 0.053},
#     {'id': 4, 'name': 'Yoshida', 'OUT': 0.651, 'BB': 0.097, '1B': 0.178, '2B': 0.05, '3B': 0.0, 'HR': 0.024},
#     {'id': 5, 'name': 'Abreu', 'OUT': 0.678, 'BB': 0.096, '1B': 0.114, '2B': 0.074, '3B': 0.004, 'HR': 0.034},
#     {'id': 6, 'name': 'Wong', 'OUT': 0.667, 'BB': 0.076, '1B': 0.179, '2B': 0.049, '3B': 0.002, 'HR': 0.027},
#     {'id': 7, 'name': 'Rafaela', 'OUT': 0.727, 'BB': 0.039, '1B': 0.159, '2B': 0.04, '3B': 0.009, 'HR': 0.026},
#     {'id': 8, 'name': 'Hamilton', 'OUT': 0.697, 'BB': 0.073, '1B': 0.148, '2B': 0.054, '3B': 0.003, 'HR': 0.025},
# ]

# SF (2024) R/G: 4.28 Simulated R/G: 4.29 / 4.287
# player_profiles = [
#     {'id': 0, 'name': 'Ramos', 'OUT': 0.679, 'BB': 0.075, '1B': 0.154, '2B': 0.044, '3B': 0.006, 'HR': 0.042},
#     {'id': 1, 'name': 'Wade', 'OUT': 0.62, 'BB': 0.165, '1B': 0.155, '2B': 0.04, '3B': 0.0, 'HR': 0.02},
#     {'id': 2, 'name': 'Chapman', 'OUT': 0.673, 'BB': 0.108, '1B': 0.114, '2B': 0.06, '3B': 0.003, 'HR': 0.042},
#     {'id': 3, 'name': 'Conforto', 'OUT': 0.691, 'BB': 0.096, '1B': 0.111, '2B': 0.055, '3B': 0.006, 'HR': 0.041},
#     {'id': 4, 'name': 'Soler', 'OUT': 0.658, 'BB': 0.134, '1B': 0.11, '2B': 0.059, '3B': 0.002, 'HR': 0.037},
#     {'id': 5, 'name': 'Yastrzemski', 'OUT': 0.698, 'BB': 0.093, '1B': 0.118, '2B': 0.034, '3B': 0.019, 'HR': 0.038},
#     {'id': 6, 'name': 'Estrada', 'OUT': 0.754, 'BB': 0.039, '1B': 0.139, '2B': 0.039, '3B': 0.005, 'HR': 0.024},
#     {'id': 7, 'name': 'Bailey', 'OUT': 0.703, 'BB': 0.087, '1B': 0.154, '2B': 0.036, '3B': 0.002, 'HR': 0.018},
#     {'id': 8, 'name': 'Fitzgerald', 'OUT': 0.666, 'BB': 0.076, '1B': 0.152, '2B': 0.056, '3B': 0.006, 'HR': 0.044},
# ]

def main():
    print("Initiate (Benchmark)...")
    simulator = BaseballSimulator(player_profiles)
    
    test_lineup = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    num_sims = 1000000
    
    # Start timing
    start_time = time.time()
    
    score, score_std = simulator.evaluate_lineup(test_lineup, num_simulations=num_sims, seed=2024)
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("-" * 40)
    print(f"Simulation : {num_sims} games")
    print(f"Total time: {elapsed_time:.4f} seconds")
    print(f"Expected score: {score:.3f} runs/game")
    print(f"Score standard deviation: {score_std:.3f} runs/game")
    print(f"Estimated 1000 MCMC steps time: {(elapsed_time * 1000) / 60:.2f} minutes")
    print("-" * 40)

if __name__ == "__main__":
    main()