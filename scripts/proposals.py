#%%
import numpy as np

def random_swap_proposal(x):
    n = len(x)
    x_prime = x.copy()
    i, j = np.random.choice(n, size=2, replace=False)
    x_prime[i], x_prime[j] = x_prime[j], x_prime[i]
    return x_prime


#%%



