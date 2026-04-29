import json

save_dir = "/home/qqaazz800624/MCMC_circular_data/results"

team = "SF"
year = 2024

player_profiles = [
    {'id': 0, 'name': 'Ramos', 'OUT': 0.679, 'BB': 0.075, '1B': 0.154, '2B': 0.044, '3B': 0.006, 'HR': 0.042},
    {'id': 1, 'name': 'Wade', 'OUT': 0.62, 'BB': 0.165, '1B': 0.155, '2B': 0.04, '3B': 0.0, 'HR': 0.02},
    {'id': 2, 'name': 'Chapman', 'OUT': 0.673, 'BB': 0.108, '1B': 0.114, '2B': 0.06, '3B': 0.003, 'HR': 0.042},
    {'id': 3, 'name': 'Conforto', 'OUT': 0.691, 'BB': 0.096, '1B': 0.111, '2B': 0.055, '3B': 0.006, 'HR': 0.041},
    {'id': 4, 'name': 'Soler', 'OUT': 0.658, 'BB': 0.134, '1B': 0.11, '2B': 0.059, '3B': 0.002, 'HR': 0.037},
    {'id': 5, 'name': 'Yastrzemski', 'OUT': 0.698, 'BB': 0.093, '1B': 0.118, '2B': 0.034, '3B': 0.019, 'HR': 0.038},
    {'id': 6, 'name': 'Estrada', 'OUT': 0.754, 'BB': 0.039, '1B': 0.139, '2B': 0.039, '3B': 0.005, 'HR': 0.024},
    {'id': 7, 'name': 'Bailey', 'OUT': 0.703, 'BB': 0.087, '1B': 0.154, '2B': 0.036, '3B': 0.002, 'HR': 0.018},
    {'id': 8, 'name': 'Fitzgerald', 'OUT': 0.666, 'BB': 0.076, '1B': 0.152, '2B': 0.056, '3B': 0.006, 'HR': 0.044},
]

with open(f"{save_dir}/player_profiles_{team}_{year}.json", "w") as f:
    json.dump(player_profiles, f, indent=4)

print(f"Player profiles for {team} {year} successfully saved to '{save_dir}/player_profiles_{team}_{year}.json'")