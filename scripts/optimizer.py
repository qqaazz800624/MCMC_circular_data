import numpy as np
from tqdm import tqdm

class MCMCOptimizer:
    def __init__(self, simulator, num_sims_per_step=1000, tau=5, initial_cache=None):
        self.simulator = simulator
        self.num_sims_per_step = num_sims_per_step
        self.tau = tau
        
        self.score_cache = initial_cache if initial_cache is not None else {}

    def _get_cached_score(self, lineup):
        lineup_tuple = tuple(lineup)
        if lineup_tuple in self.score_cache:
            return self.score_cache[lineup_tuple]
        
        score = self.simulator.evaluate_lineup(list(lineup), num_simulations=self.num_sims_per_step)
        self.score_cache[lineup_tuple] = score
        return score

    def optimize(self, initial_x, proposal_func, max_steps=5000, seed=None):
        
        rng = np.random.default_rng(seed)
        
        current_x = initial_x.copy()
        current_F = self._get_cached_score(current_x)
        
        best_x = current_x.copy()
        best_F = current_F
        step_when_best_found = 0

        visited_states = set()
        visited_states.add(tuple(current_x))

        score_history = [current_F]
        state_history = [current_x.copy()]

        improvement_log = [{"step": 0, "score": float(best_F), "note": "Initial"}]
        
        for t in tqdm(range(max_steps), desc="Running MCMC Steps", leave=False):
            actual_step = t + 1 
            proposed_x = proposal_func(current_x, rng)

            visited_states.add(tuple(proposed_x))

            proposed_F = self._get_cached_score(proposed_x)
            delta_F = proposed_F - current_F
            
            if delta_F > 0:
                acceptance_prob = 1.0
            else:
                acceptance_prob = np.exp(max(delta_F / self.tau, -700))
                
            if rng.random() <= acceptance_prob:
                current_x = proposed_x
                current_F = proposed_F
                
            if current_F > best_F + 1e-6:
                best_F = current_F
                best_x = current_x.copy()
                step_when_best_found = actual_step
                improvement_log.append({"step": actual_step, "score": float(best_F), "note": "New High"})

            elif abs(current_F - best_F) <= 1e-6:
                if tuple(current_x) != tuple(best_x):
                    best_x = current_x.copy()
                    improvement_log.append({"step": actual_step, "score": float(current_F), "note": "Tie"})
                else:
                    improvement_log.append({"step": actual_step, "score": float(current_F), "note": "Revisit Best"})

            score_history.append(current_F)
            state_history.append(current_x.copy())
                    
        return {
            "initial_state": initial_x,
            "best_lineup": best_x,
            "best_score": best_F,
            "steps_to_best": step_when_best_found,
            "total_steps": max_steps,
            "visited_states": visited_states,
            "score_history": score_history,
            "state_history": state_history,
            "improvement_log": improvement_log
        }