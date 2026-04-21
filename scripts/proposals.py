#%%
import numpy as np

def random_swap_proposal(x):
    n = len(x)
    x_prime = x.copy()
    i, j = np.random.choice(n, size=2, replace=False)
    x_prime[i], x_prime[j] = x_prime[j], x_prime[i]
    return x_prime


def random_insertion_proposal(x):
    n = len(x)
    i, j = np.random.choice(n, size=2, replace=False)
    elem = x[i]
    x_prime = np.delete(x, i)
    x_prime = np.insert(x_prime, j, elem)
    return x_prime


def directional_reversal_proposal(x):
    n = len(x)
    x_prime = np.asarray(x).copy()
    i, j = np.random.choice(n, size=2, replace=False)
    
    if i < j:
        x_prime[i:j+1] = x_prime[i:j+1][::-1]
    else:
        B = np.concatenate((x_prime[i:], x_prime[:j+1]))
        B_rev = B[::-1]
        tail_len = n - i
        x_prime[i:] = B_rev[:tail_len]
        x_prime[:j+1] = B_rev[tail_len:]
        
    return x_prime


def k_cycle_shift_proposal(x, k=3):
    n = len(x)
    x_prime = np.asarray(x).copy()
    
    if k is None:
        k = np.random.randint(2, n + 1)
        
    I = np.sort(np.random.choice(n, size=k, replace=False))
    direction = np.random.choice(['forward', 'backward'])
    V = x_prime[I]

    if direction == 'backward':
        V_prime = np.roll(V, shift=1)
    else:
        V_prime = np.roll(V, shift=-1)
        
    x_prime[I] = V_prime
    
    return x_prime


def block_pair_exchange_proposal(x):
    n = len(x)
    x_prime = np.asarray(x).copy()
    c1, c2, c3 = np.sort(np.random.choice(n, size=3, replace=False))
    
    B1 = x_prime[c1:c2]
    B2 = x_prime[c2:c3]
    B3 = np.concatenate((x_prime[c3:], x_prime[:c1]))
    option = np.random.choice([1, 2, 3])
    
    if option == 1:
        S = np.concatenate((B2, B1, B3))
    elif option == 2:
        S = np.concatenate((B1, B3, B2))
    else:
        S = np.concatenate((B3, B2, B1))
        
    h = c1 
    x_prime = np.roll(S, shift=h)
    
    return x_prime

def hybrid_swap_insertion_proposal(x):
    if np.random.rand() < 0.5:
        return random_swap_proposal(x)
    else:
        return random_insertion_proposal(x)

def hybrid_swap_reversal_proposal(x):
    if np.random.rand() < 0.5:
        return random_swap_proposal(x)
    else:
        return directional_reversal_proposal(x)

def hybrid_swap_kcycle_proposal(x, k=None):
    if np.random.rand() < 0.5:
        return random_swap_proposal(x)
    else:
        return k_cycle_shift_proposal(x, k)

def hybrid_swap_block_exchange_proposal(x):
    if np.random.rand() < 0.5:
        return random_swap_proposal(x)
    else:
        return block_pair_exchange_proposal(x)

def hybrid_insertion_reversal_proposal(x):
    if np.random.rand() < 0.5:
        return random_insertion_proposal(x)
    else:
        return directional_reversal_proposal(x)

def hybrid_insertion_kcycle_proposal(x, k=None):
    if np.random.rand() < 0.5:
        return random_insertion_proposal(x)
    else:
        return k_cycle_shift_proposal(x, k)

def hybrid_insertion_block_exchange_proposal(x):
    if np.random.rand() < 0.5:
        return random_insertion_proposal(x)
    else:
        return block_pair_exchange_proposal(x)

def hybrid_reversal_kcycle_proposal(x, k=None):
    if np.random.rand() < 0.5:
        return directional_reversal_proposal(x)
    else:
        return k_cycle_shift_proposal(x, k)

def hybrid_reversal_block_exchange_proposal(x):
    if np.random.rand() < 0.5:
        return directional_reversal_proposal(x)
    else:
        return block_pair_exchange_proposal(x)

def hybrid_kcycle_block_exchange_proposal(x, k=None):
    if np.random.rand() < 0.5:
        return k_cycle_shift_proposal(x, k)
    else:
        return block_pair_exchange_proposal(x)

#%%





#%%



