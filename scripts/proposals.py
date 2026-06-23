import numpy as np

def random_swap_proposal(x, rng):
    n = len(x)
    x_prime = x.copy()
    i, j = rng.choice(n, size=2, replace=False)
    x_prime[i], x_prime[j] = x_prime[j], x_prime[i]
    return x_prime


def random_insertion_proposal(x, rng):
    n = len(x)
    i, j = rng.choice(n, size=2, replace=False)
    elem = x[i]
    x_prime = np.delete(x, i)
    x_prime = np.insert(x_prime, j, elem)
    return x_prime


def directional_reversal_proposal(x, rng):
    n = len(x)
    x_prime = np.asarray(x).copy()
    i, j = rng.choice(n, size=2, replace=False)
    
    if i < j:
        x_prime[i:j+1] = x_prime[i:j+1][::-1]
    else:
        B = np.concatenate((x_prime[i:], x_prime[:j+1]))
        B_rev = B[::-1]
        tail_len = n - i
        x_prime[i:] = B_rev[:tail_len]
        x_prime[:j+1] = B_rev[tail_len:]
        
    return x_prime

def k_cycle_shift_proposal(x, rng, k=None):
    n = len(x)
    x_prime = np.asarray(x).copy()
    
    if k is None:
        k = rng.integers(2, n + 1)
        
    I = np.sort(rng.choice(n, size=k, replace=False))
    direction = rng.choice(['forward', 'backward'])
    
    V = x_prime[I]

    if direction == 'backward':
        V_prime = np.roll(V, shift=1)
    else:
        V_prime = np.roll(V, shift=-1)
        
    x_prime[I] = V_prime
    
    return x_prime

def block_pair_exchange_proposal(x, rng):
    n = len(x)
    x_prime = np.asarray(x).copy()
    c1, c2, c3 = np.sort(rng.choice(n, size=3, replace=False))
    
    B1 = x_prime[c1:c2]
    B2 = x_prime[c2:c3]
    B3 = np.concatenate((x_prime[c3:], x_prime[:c1]))
    
    option = rng.choice([1, 2, 3])
    
    if option == 1:
        S = np.concatenate((B2, B1, B3))
    elif option == 2:
        S = np.concatenate((B1, B3, B2))
    else:
        S = np.concatenate((B3, B2, B1))
        
    h = c1 
    x_prime = np.roll(S, shift=h)
    
    return x_prime

def hybrid_swap_insertion_proposal(x, rng):
    if rng.random() < 0.5:
        return random_swap_proposal(x, rng)
    else:
        return random_insertion_proposal(x, rng)

def hybrid_swap_reversal_proposal(x, rng):
    if rng.random() < 0.5:
        return random_swap_proposal(x, rng)
    else:
        return directional_reversal_proposal(x, rng)

def hybrid_swap_kcycle_proposal(x, rng, k=None):
    if rng.random() < 0.5:
        return random_swap_proposal(x, rng)
    else:
        return k_cycle_shift_proposal(x, rng, k)

def hybrid_swap_block_exchange_proposal(x, rng):
    if rng.random() < 0.5:
        return random_swap_proposal(x, rng)
    else:
        return block_pair_exchange_proposal(x, rng)

def hybrid_insertion_reversal_proposal(x, rng):
    if rng.random() < 0.5:
        return random_insertion_proposal(x, rng)
    else:
        return directional_reversal_proposal(x, rng)

def hybrid_insertion_kcycle_proposal(x, rng, k=None):
    if rng.random() < 0.5:
        return random_insertion_proposal(x, rng)
    else:
        return k_cycle_shift_proposal(x, rng, k)

def hybrid_insertion_block_exchange_proposal(x, rng):
    if rng.random() < 0.5:
        return random_insertion_proposal(x, rng)
    else:
        return block_pair_exchange_proposal(x, rng)

def hybrid_reversal_kcycle_proposal(x, rng, k=None):
    if rng.random() < 0.5:
        return directional_reversal_proposal(x, rng)
    else:
        return k_cycle_shift_proposal(x, rng, k)

def hybrid_reversal_block_exchange_proposal(x, rng):
    if rng.random() < 0.5:
        return directional_reversal_proposal(x, rng)
    else:
        return block_pair_exchange_proposal(x, rng)

def hybrid_kcycle_block_exchange_proposal(x, rng, k=None):
    if rng.random() < 0.5:
        return k_cycle_shift_proposal(x, rng, k)
    else:
        return block_pair_exchange_proposal(x, rng)

import numpy as np

def local_biased_circular_insertion(x, rng):
    """
    Local-Biased Circular Insertion (LBCI)
    A decoupled, three-stage proposal function combining cyclic shifting, 
    local fine-tuning, and global exploration for circular data optimization.
    """
    n = len(x)
    
    # 1. Macro-level (10% probability): Cyclic Shift
    # Preserves all relative adjacent synergies perfectly. Shifts the entire 
    # sequence to optimize the absolute positioning (e.g., Plate Appearances).
    if rng.random() < 0.10:
        shift = rng.integers(1, n)
        return np.roll(x, shift)
        
    # Prepare for the insertion operation
    x_prime = np.asarray(x).copy()
    i = rng.integers(0, n)
    elem = x_prime[i]
    
    # 2. Micro-level (~72% probability): Local-biased Insertion
    # Designed for the late stages of MCMC. Restricts movement to 1-2 positions 
    # adjacent to the original spot. Maintains a high acceptance rate and 
    # accelerates convergence to the global maximum.
    if rng.random() < 0.80:
        shift = rng.choice([-2, -1, 1, 2])
        j = (i + shift) % n  # Perfectly supports the circular wrap-around 
    
    # 3. Meso-level (~18% probability): Global Random Insertion
    # Retains the exploration capability of standard Random Insertion 
    # to prevent the chain from being trapped in local optima.
    else:
        j = rng.integers(0, n)
        
    # Prevent invalid in-place operations that waste iterations
    if j == i:
        j = (i + 1) % n
        
    # Execute removal and insertion
    x_prime = np.delete(x_prime, i)
    x_prime = np.insert(x_prime, j, elem)
    
    return x_prime

def von_mises_fisher_circular_insertion(x, rng, kappa=1.5):
    """
    von Mises-Fisher Circular Insertion (VMFCI) Proposal
    Inspired by Sinusoidal Positional Embedding, this proposal utilizes the circular normal 
    distribution (von Mises-Fisher distribution) to calculate transition probabilities based on 
    relative positions, integrating local fine-tuning with global exploration.
    """
    n = len(x)
    
    # 1. Macro-level: Retain cyclic shifts to optimize the absolute positional distribution
    if rng.random() < 0.10:
        shift = rng.integers(1, n)
        return np.roll(x, shift)
        
    x_prime = np.asarray(x).copy()
    i = rng.integers(0, n)
    elem = x_prime[i]
    
    # 2. von Mises-Fisher probability calculation
    # Calculate the angular differences between all positions (0 to n-1) and the origin i
    positions = np.arange(n)
    angles = 2 * np.pi * (positions - i) / n
    
    # Apply the cosine function to compute the unnormalized weights
    weights = np.exp(kappa * np.cos(angles))
    
    # To ensure a strictly new state transition, set the probability of staying in place to 0
    weights[i] = 0
    
    # Normalize the weights to form a valid probability distribution (sum to 1)
    probs = weights / np.sum(weights)
    
    # Sample the target insertion index j based on the computed sinusoidal probabilities
    j = rng.choice(positions, p=probs)
    
    # Execute the element removal and insertion operations
    x_prime = np.delete(x_prime, i)
    x_prime = np.insert(x_prime, j, elem)
    
    return x_prime