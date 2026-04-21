#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = '/home/qqaazz800624/MCMC_circular_data/results'
proposal = 'random_swap'

filename = os.path.join(data_dir, f'chain_details_{proposal}_proposal_tau5.0.csv')
df = pd.read_csv(filename)
hitting_times = df['hitting_time'].dropna().values
x = np.sort(hitting_times)
y = np.arange(1, len(x) + 1) / len(x)

plt.figure(figsize=(10, 6))
plt.step(x, y, where='post', color='royalblue', linewidth=2)

plt.title(f'Empirical CDF of Hitting Times ({proposal.replace("_", " ").title()}, $\\tau=5.0$)', fontsize=14)
plt.xlabel('Hitting Time (steps)', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

#plt.show()
plt.savefig(os.path.join(data_dir, f'ecdf_hitting_times_{proposal}.png'), dpi=300) 


#%%





#%%






#%%