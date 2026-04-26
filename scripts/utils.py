import numpy as np
import matplotlib.pyplot as plt
import os
import objectives

def plot_cdfs(sorted_scores, probabilities, tau, 
              save_dir="results",
              save_path="distribution_cdfs.png"):
    N = len(sorted_scores)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ascending_scores = sorted_scores[::-1] 
    cdf_raw = np.arange(1, N + 1) / N
    ax1.plot(ascending_scores, cdf_raw, color='royalblue', linewidth=2)
    ax1.set_title('CDF of Original Objective Function $F(x)$')
    ax1.set_xlabel('F(x) Score')
    ax1.set_ylabel('Cumulative Probability')
    ax1.grid(True, alpha=0.3)

    top_k = min(100, N)
    cdf_boltzmann = np.cumsum(probabilities[:top_k])
    ax2.plot(range(1, top_k + 1), cdf_boltzmann, color='crimson', linewidth=2)
    ax2.set_title(f'CDF of Boltzmann Prob. (Top {top_k}, $\\tau={tau}$)')
    ax2.set_xlabel(f'Rank (1 to {top_k}, sorted by score)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, save_path), dpi=300)
    #plt.show()

def generate_initial_states(elements, num_states=10000, seed=42):
    """
    Pre-generates a fixed set of initial states to ensure an identical starting point (fair baseline) for all methods.
    
    Parameters:
    -----------
    elements : list or np.ndarray
        The base elements (building blocks) to be permuted.
    num_states : int
        The number of initial states to generate.
    seed : int
        The random seed for reproducibility.
    """
    np.random.seed(seed)
    
    elements = np.asarray(elements) 
    initial_states = []
    
    for _ in range(num_states):
        state = elements.copy()
        np.random.shuffle(state)
        initial_states.append(state)
        
    return initial_states


import numpy as np

def run_single_mcmc_chain(initial_x, 
                          objective_func, 
                          proposal_func, 
                          g, 
                          alpha, 
                          beta, tau, 
                          true_max_F, 
                          max_steps=50000):
    """
    Run a single MCMC chain starting from initial_x.
    Returns: (final_x, final_F, hitting_time)
    """
    current_x = initial_x.copy()
    current_F = objective_func(current_x, g, alpha, beta)[0]

    if current_F >= true_max_F - 1e-4: 
        return current_x, current_F, 0 

    for t in range(1, max_steps + 1):
        proposed_x = proposal_func(current_x)
        proposed_F = objective_func(proposed_x, g, alpha, beta)[0]
        delta_F = proposed_F - current_F

        if delta_F > 0:
            acceptance_prob = 1.0
        else:
            acceptance_prob = np.exp(delta_F / tau)

        if np.random.rand() <= acceptance_prob:
            current_x = proposed_x
            current_F = proposed_F

        if current_F >= true_max_F - 1e-4:
            return current_x, current_F, t 

    return current_x, current_F, max_steps


def run_baseball_mcmc(initial_x, 
                      proposal_func, 
                      num_sims_per_step, 
                      max_steps, 
                      tau):
    current_x = initial_x.copy()
    current_F = objectives.objective_baseball_sim(current_x, prob_matrix, num_simulations=num_sims_per_step)
    
    best_x = current_x.copy()
    best_F = current_F
    
    for t in tqdm(range(max_steps), desc="MCMC Optimizing"):
        proposed_x = proposal_func(current_x)
        proposed_F = objectives.objective_baseball_sim(proposed_x, prob_matrix, num_simulations=num_sims_per_step)
        
        delta_F = proposed_F - current_F
        
        if delta_F > 0:
            acceptance_prob = 1.0
        else:
            acceptance_prob = np.exp(delta_F / tau)
            
        if np.random.rand() <= acceptance_prob:
            current_x = proposed_x
            current_F = proposed_F
            
            if current_F > best_F:
                best_F = current_F
                best_x = current_x.copy()
                
    return best_x, best_F