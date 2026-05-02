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
            raise ValueError(f"The length of weight array g ({len(g)}) "
                             f"must match the length of input array x ({n}).")

    linear_term = np.sum(g * x)
    
    interaction_term = 0
    for i in range(n):
        indices = [(i + j) % n for j in range(5)]
        interaction_term += np.prod(x[indices])

    out = (alpha * linear_term + beta * interaction_term-36)*100 - 14570
        
    return out, linear_term, interaction_term


class BaseballSimulator:
    def __init__(self, player_profiles):
        self.event_types = ['OUT', 'BB', '1B', '2B', '3B', 'HR']
        self.player_profiles = player_profiles
        
        self.prob_matrix = np.array([[p[event] for event in self.event_types] for p in player_profiles])
        self.name_dict = {p['id']: p['name'] for p in player_profiles}
        self.prob_1b_to_home_on_double = 0.40  
        self.prob_2b_to_home_on_single = 0.60

    def _simulate_single_game(self, lineup_indices, rng, innings=9):
        total_runs = 0
        current_batter_idx = 0
        lineup_probs = self.prob_matrix[lineup_indices]
        
        for _ in range(innings):
            outs = 0
            bases = np.array([0, 0, 0]) 
            
            while outs < 3:
                probs = lineup_probs[current_batter_idx]
                
                result = rng.choice(self.event_types, p=probs)
                
                if result == 'OUT':
                    outs += 1
                elif result == 'HR':
                    total_runs += np.sum(bases) + 1
                    bases = np.array([0, 0, 0])
                elif result == '3B':
                    total_runs += np.sum(bases)
                    bases = np.array([0, 0, 1])
                elif result == '2B':
                    total_runs += bases[1] + bases[2]
                    if bases[0] == 1:
                        if rng.random() < self.prob_1b_to_home_on_double:
                            total_runs += 1
                            bases = np.array([0, 1, 0])
                        else:
                            bases = np.array([0, 1, 1]) 
                    else:
                        bases = np.array([0, 1, 0])
                elif result == '1B':
                    total_runs += bases[2]
                    
                    if bases[1] == 1:
                        if rng.random() < self.prob_2b_to_home_on_single:
                            total_runs += 1           
                            bases = np.array([1, bases[0], 0]) 
                        else:
                            bases = np.array([1, bases[0], 1]) 
                    else:
                        bases = np.array([1, bases[0], 0])
                elif result == 'BB':
                    if bases[0] == 1 and bases[1] == 1 and bases[2] == 1:
                        total_runs += 1
                    elif bases[0] == 1 and bases[1] == 1:
                        bases = np.array([1, 1, 1])
                    elif bases[0] == 1:
                        bases = np.array([1, 1, bases[2]])
                    else:
                        bases = np.array([1, bases[1], bases[2]])
                
                current_batter_idx = (current_batter_idx + 1) % 9
                
        return total_runs

    def evaluate_lineup(self, lineup_indices, num_simulations=1000, seed=42):
        rng = np.random.default_rng(seed)
        runs = sum(self._simulate_single_game(lineup_indices, rng) for _ in range(num_simulations))
        #runs_std = np.std([self._simulate_single_game(lineup_indices, rng) for _ in range(num_simulations)])/np.sqrt(num_simulations)
        
        return runs / num_simulations#, runs_std
