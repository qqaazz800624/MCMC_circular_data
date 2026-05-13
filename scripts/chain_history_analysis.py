#%%

import numpy as np

data = np.load("results/mcmc_traces_1st_1000steps_LAD_2024_random_swap_proposal_1chains.npz")
scores = data['scores']
states = data['states']

print(f"First chain, step 500: {states[0][500]}")
print(f"Expected score at that step: {scores[0][500]}")


#%%





#%%






#%%