#%%

import numpy as np

file_path = "/home/qqaazz800624/MCMC_circular_data/results/mcmc_traces_LAD_2024_k_cycle_shift_proposal_1chains.npy"

traces = np.load(file_path)
all_scores = traces.flatten()
unique_scores = np.unique(all_scores)
sorted_scores = np.sort(unique_scores)[::-1]

print("=" * 50)
print(f"Analyzing: {file_path.split('/')[-1]}")
print("=" * 50)

if len(sorted_scores) >= 2:
    highest_score = sorted_scores[0]
    second_highest_score = sorted_scores[1]
    diff = highest_score - second_highest_score
    
    print(f"Highest score: {highest_score:.4f}")
    print(f"Second highest score: {second_highest_score:.4f}")
    print(f"Difference: {diff:.4f}")
    print("-" * 50)
    
    count_highest = np.sum(all_scores == highest_score)
    print(f"Highest score appeared {count_highest} times in the trace")

elif len(sorted_scores) == 1:
    print(f"Only one unique score in the entire trace: {sorted_scores[0]:.3f}")
else:
    print("No score data found in the file!")
print("=" * 50)

#%%





#%%






#%%