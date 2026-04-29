import numpy as np
from tqdm import tqdm

class MCMCOptimizer:
    def __init__(self, simulator, num_sims_per_step=1000, tau=5):
        self.simulator = simulator
        self.num_sims_per_step = num_sims_per_step
        self.tau = tau
        
        self.score_cache = {}

    def _get_cached_score(self, lineup):
        lineup_tuple = tuple(lineup)
        if lineup_tuple in self.score_cache:
            return self.score_cache[lineup_tuple]
        
        score = self.simulator.evaluate_lineup(list(lineup), num_simulations=self.num_sims_per_step)
        self.score_cache[lineup_tuple] = score
        return score

    def optimize(self, initial_x, proposal_func, max_steps=5000):
        current_x = initial_x.copy()
        current_F = self._get_cached_score(current_x)
        
        best_x = current_x.copy()
        best_F = current_F
        step_when_best_found = 0
        
        for t in tqdm(range(max_steps), desc="Running MCMC Steps", leave=False):
            proposed_x = proposal_func(current_x)
            proposed_F = self._get_cached_score(proposed_x)
            
            delta_F = proposed_F - current_F
            
            if delta_F > 0:
                acceptance_prob = 1.0
            else:
                acceptance_prob = np.exp(max(delta_F / self.tau, -700))
                
            if np.random.rand() <= acceptance_prob:
                current_x = proposed_x
                current_F = proposed_F
                
                if current_F > best_F:
                    best_F = current_F
                    best_x = current_x.copy()
                    step_when_best_found = t
                    
        return {
            "initial_state": initial_x,
            "best_lineup": best_x,
            "best_score": best_F,
            "steps_to_best": step_when_best_found,
            "total_steps": max_steps
        }