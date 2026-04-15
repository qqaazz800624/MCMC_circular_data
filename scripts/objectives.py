#%%

import numpy as np

def toy_objective_1(x, g=None, alpha=1.0, beta=0.05):
    x = np.asarray(x)
    n = len(x)
    
    if g is None:
        g = np.arange(n, 0, -1)
    else:
        g = np.asarray(g)
        if len(g) != n:
            raise ValueError(f"The length of weight array g ({len(g)}) must match the length of input array x ({n}).")

    linear_term = np.sum(g * x)
    
    interaction_term = 0
    for i in range(n):
        indices = [(i + j) % n for j in range(5)]
        interaction_term += np.prod(x[indices])

    out = (alpha * linear_term + beta * interaction_term-36)*100 - 775
        
    return out, linear_term, interaction_term


#%%


