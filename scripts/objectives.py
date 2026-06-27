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
        
        return runs / num_simulations #, runs_std


class BaseballModel:
    def __init__(self, player_profiles):
        """
        Initializes the analytical engine for baseball lineups using Bukiet (1997)
        with support for multi-out granular sub-classification.
        
        player_profiles: List of dicts matching your fine-grained JSON format.
        """
        self.event_types = ['SINGLE_OUT', 'DOUBLE_OUT', 'TRIPLE_OUT', 'BB', '1B', '2B', '3B', 'HR']
        self.player_profiles = player_profiles
        
        # Numerical probability matrices mapped to profile arrays
        self.prob_matrix = np.array([[p[event] for event in self.event_types] for p in player_profiles])
        self.name_dict = {p['id']: p['name'] for p in player_profiles}
        
        # Baserunning transition parameters shared across both engines
        self.prob_1b_to_home_on_double = 0.40  
        self.prob_2b_to_home_on_single = 0.60
        
        # Build state definitions for the 25-state Markov framework (Bukiet 1997)
        self.base_configurations = ["Empty", "1B", "2B", "3B", "1B_2B", "1B_3B", "2B_3B", "Full"]
        self.state_to_idx = {}
        state_counter = 0
        for outs in [0, 1, 2]:
            for base in self.base_configurations:
                self.state_to_idx[f"{outs}Outs_{base}"] = state_counter
                state_counter += 1
        self.state_to_idx["3Outs"] = 24

    def _advance_runners_analytically(self, current_base: str, hit_type: str) -> dict:
        """
        Calculates all possible destination base configurations and run rewards 
        with their associated probabilities matching the simulation specifications.
        """
        if hit_type == "HR":
            return {("Empty", 1 + len([b for b in current_base.split('_') if b in ["1B", "2B", "3B"]])): 1.0}
        if hit_type == "3B":  
            return {("3B", len([b for b in current_base.split('_') if b in ["1B", "2B", "3B"]])): 1.0}
        
        if hit_type == "BB":
            rules = {
                "Empty": ("1B", 0), "1B": ("1B_2B", 0), "2B": ("1B_2B", 0), "3B": ("1B_3B", 0),
                "1B_2B": ("Full", 0), "1B_3B": ("Full", 0), "2B_3B": ("Full", 0), "Full": ("Full", 1)
            }
            return {rules[current_base]: 1.0}

        if hit_type == "1B":  
            if current_base == "Empty": return {("1B", 0): 1.0}
            if current_base == "1B": return {("1B_2B", 0): 1.0}
            if current_base == "3B": return {("1B", 1): 1.0}
            if current_base == "1B_3B": return {("1B_2B", 1): 1.0}
            
            p_score = self.prob_2b_to_home_on_single
            if current_base == "2B": return {("1B", 1): p_score, ("1B_3B", 0): 1.0 - p_score}
            if current_base == "1B_2B": return {("1B_3B", 1): p_score, ("Full", 0): 1.0 - p_score} 
            if current_base == "2B_3B": return {("1B", 2): p_score, ("1B_3B", 1): 1.0 - p_score}
            if current_base == "Full": return {("1B_3B", 2): p_score, ("Full", 1): 1.0 - p_score}

        if hit_type == "2B":  
            p_score = self.prob_1b_to_home_on_double
            if current_base == "Empty": return {("2B", 0): 1.0}
            if current_base == "2B": return {("2B", 1): 1.0}
            if current_base == "3B": return {("2B", 1): 1.0}
            if current_base == "2B_3B": return {("2B", 2): 1.0}
            
            if current_base == "1B": return {("2B", 1): p_score, ("2B_3B", 0): 1.0 - p_score}
            if current_base == "1B_2B": return {("2B", 2): p_score, ("2B_3B", 1): 1.0 - p_score}
            if current_base == "1B_3B": return {("2B", 2): p_score, ("2B_3B", 1): 1.0 - p_score}
            if current_base == "Full": return {("2B", 3): p_score, ("2B_3B", 2): 1.0 - p_score}

        return {("Empty", 0): 1.0}

    def _compile_lineup_matrices(self, lineup_indices) -> list:
        """Compiles the 25x25 Markov transition matrices for the 9 selected lineup elements."""
        lineup_matrices = []
        for idx in lineup_indices:
            probs = dict(zip(self.event_types, self.prob_matrix[idx]))
            P = np.zeros((25, 25))
            R_expected = np.zeros((25, 25))  
            
            for outs in [0, 1, 2]:
                for base in self.base_configurations:
                    current_state = f"{outs}Outs_{base}"
                    from_idx = self.state_to_idx[current_state]
                    
                    # 1. SINGLE_OUT (精確增加 1 個出局數)
                    p_s_out = probs["SINGLE_OUT"]
                    if p_s_out > 0:
                        if outs == 2:
                            P[from_idx, 24] += p_s_out
                        else:
                            P[from_idx, self.state_to_idx[f"{outs + 1}Outs_{base}"]] += p_s_out
                    
                    # 2. DOUBLE_OUT (修復 KeyError 邊界條件)
                    p_d_out = probs["DOUBLE_OUT"]
                    if p_d_out > 0:
                        has_force_at_1b = "1B" in base or base == "Full"
                        if outs < 2 and has_force_at_1b:
                            next_outs = outs + 2
                            
                            # 如果加完 2 出局後達到或超過 3 出局，直接送入 24 號吸收態 ("3Outs")
                            if next_outs >= 3:
                                P[from_idx, 24] += p_d_out
                            else:
                                dp_base_mapping = {"1B": "Empty", "1B_2B": "2B", "1B_3B": "3B", "Full": "2B_3B"}
                                target_base = dp_base_mapping[base]
                                P[from_idx, self.state_to_idx[f"{next_outs}Outs_{target_base}"]] += p_d_out
                        else:
                            # 不滿足雙殺條件，退化為普通一出局
                            if outs == 2:
                                P[from_idx, 24] += p_d_out
                            else:
                                P[from_idx, self.state_to_idx[f"{outs + 1}Outs_{base}"]] += p_d_out

                    # 3. TRIPLE_OUT (直接無條件結束半局)
                    p_t_out = probs["TRIPLE_OUT"]
                    if p_t_out > 0:
                        P[from_idx, 24] += p_t_out
                    
                    # 4. 安全上壘與保送事件 (1B, 2B, 3B, HR, BB)
                    for hit_type in ["1B", "2B", "3B", "HR", "BB"]:
                        prob = probs[hit_type]
                        if prob == 0: continue
                        
                        branches = self._advance_runners_analytically(base, hit_type)
                        for (new_base, runs), weight in branches.items():
                            to_idx = self.state_to_idx[f"{outs}Outs_{new_base}"]
                            
                            transition_prob = prob * weight
                            P[from_idx, to_idx] += transition_prob
                            R_expected[from_idx, to_idx] += transition_prob * runs
                            
            P[24, 24] = 1.0
            lineup_matrices.append((P, R_expected))
        return lineup_matrices

    def evaluate_lineup_analytically(self, lineup_indices, innings=9) -> float:
        """
        [F(x)] Evaluates lineup expected score with zero variance using linear arithmetic.
        Execution speed is extremely fast (<1 millisecond).
        """
        lineup_matrices = self._compile_lineup_matrices(lineup_indices)
        u = np.zeros(25)
        u[0] = 1.0
        total_expected_runs = 0.0
        batter_idx = 0
        
        while u[24] < 1.0 - 1e-6:
            P_b, R_expected_b = lineup_matrices[batter_idx]
            current_expected_runs = np.sum(u[:, None] * R_expected_b)
            total_expected_runs += current_expected_runs
            
            u = u @ P_b
            batter_idx = (batter_idx + 1) % 9
            
        return total_expected_runs * innings
