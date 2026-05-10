import json

save_dir = "/home/qqaazz800624/MCMC_circular_data/results"

team = "LAA"
year = 2023

player_profiles = [
    {'id': 0, 'name': 'Ward', 'OUT': 0.665, 'BB': 0.115, '1B': 0.142, '2B': 0.044, '3B': 0.0, 'HR': 0.034},
    {'id': 1, 'name': 'Ohtani', 'OUT': 0.584, 'BB': 0.165, '1B': 0.122, '2B': 0.043, '3B': 0.013, 'HR': 0.073},
    {'id': 2, 'name': 'Trout', 'OUT': 0.631, 'BB': 0.144, '1B': 0.133, '2B': 0.039, '3B': 0.003, 'HR': 0.05},
    {'id': 3, 'name': 'Drury', 'OUT': 0.694, 'BB': 0.063, '1B': 0.13, '2B': 0.057, '3B': 0.006, 'HR': 0.05},
    {'id': 4, 'name': 'Renfroe', 'OUT': 0.702, 'BB': 0.086, '1B': 0.119, '2B': 0.057, '3B': 0.0, 'HR': 0.036},
    {'id': 5, 'name': 'Rengifo', 'OUT': 0.66, 'BB': 0.106, '1B': 0.155, '2B': 0.034, '3B': 0.009, 'HR': 0.036},
    {'id': 6, 'name': "O'Hoppe", 'OUT': 0.704, 'BB': 0.08, '1B': 0.116, '2B': 0.03, '3B': 0.0, 'HR': 0.07},
    {'id': 7, 'name': 'Moniak', 'OUT': 0.694, 'BB': 0.037, '1B': 0.155, '2B': 0.065, '3B': 0.006, 'HR': 0.043},
    {'id': 8, 'name': 'Neto', 'OUT': 0.693, 'BB': 0.109, '1B': 0.119, '2B': 0.052, '3B': 0.0, 'HR': 0.027}
]

with open(f"{save_dir}/player_profiles_{team}_{year}.json", "w") as f:
    json.dump(player_profiles, f, indent=4)

print(f"Player profiles for {team} {year} successfully saved to '{save_dir}/player_profiles_{team}_{year}.json'")