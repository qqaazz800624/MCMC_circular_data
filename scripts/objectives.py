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


# class BaseballSimulator:
#     def __init__(self, player_profiles):
#         self.event_types = ['OUT', 'BB', '1B', '2B', '3B', 'HR']
#         self.player_profiles = player_profiles
        
#         self.prob_matrix = np.array([[p[event] for event in self.event_types] for p in player_profiles])
#         self.name_dict = {p['id']: p['name'] for p in player_profiles}
#         self.prob_1b_to_home_on_double = 0.40  
#         self.prob_2b_to_home_on_single = 0.60

#     def _simulate_single_game(self, lineup_indices, rng, innings=9):
#         total_runs = 0
#         current_batter_idx = 0
#         lineup_probs = self.prob_matrix[lineup_indices]
        
#         for _ in range(innings):
#             outs = 0
#             bases = np.array([0, 0, 0]) 
            
#             while outs < 3:
#                 probs = lineup_probs[current_batter_idx]
                
#                 result = rng.choice(self.event_types, p=probs)
                
#                 if result == 'OUT':
#                     outs += 1
#                 elif result == 'HR':
#                     total_runs += np.sum(bases) + 1
#                     bases = np.array([0, 0, 0])
#                 elif result == '3B':
#                     total_runs += np.sum(bases)
#                     bases = np.array([0, 0, 1])
#                 elif result == '2B':
#                     total_runs += bases[1] + bases[2]
#                     if bases[0] == 1:
#                         if rng.random() < self.prob_1b_to_home_on_double:
#                             total_runs += 1
#                             bases = np.array([0, 1, 0])
#                         else:
#                             bases = np.array([0, 1, 1]) 
#                     else:
#                         bases = np.array([0, 1, 0])
#                 elif result == '1B':
#                     total_runs += bases[2]
                    
#                     if bases[1] == 1:
#                         if rng.random() < self.prob_2b_to_home_on_single:
#                             total_runs += 1           
#                             bases = np.array([1, bases[0], 0]) 
#                         else:
#                             bases = np.array([1, bases[0], 1]) 
#                     else:
#                         bases = np.array([1, bases[0], 0])
#                 elif result == 'BB':
#                     if bases[0] == 1 and bases[1] == 1 and bases[2] == 1:
#                         total_runs += 1
#                     elif bases[0] == 1 and bases[1] == 1:
#                         bases = np.array([1, 1, 1])
#                     elif bases[0] == 1:
#                         bases = np.array([1, 1, bases[2]])
#                     else:
#                         bases = np.array([1, bases[1], bases[2]])
                
#                 current_batter_idx = (current_batter_idx + 1) % 9
                
#         return total_runs

#     def evaluate_lineup(self, lineup_indices, num_simulations=1000, seed=42):
#         rng = np.random.default_rng(seed)
#         runs = sum(self._simulate_single_game(lineup_indices, rng) for _ in range(num_simulations))
#         #runs_std = np.std([self._simulate_single_game(lineup_indices, rng) for _ in range(num_simulations)])/np.sqrt(num_simulations)
        
#         return runs / num_simulations #, runs_std

# class BaseballSimulator:
#     """
#     舊版 Monte Carlo simulator (事件: OUT/BB/1B/2B/3B/HR, 無雙殺)。
#     向量化以支援大樣本; evaluate_lineup 回傳 (平均得分, 單場得分標準差, 標準誤)。
#     """
#     EVENTS = ['OUT', 'BB', '1B', '2B', '3B', 'HR']

#     def __init__(self, player_profiles):
#         self.event_types = self.EVENTS
#         self.player_profiles = player_profiles
#         self.prob_matrix = np.array([[p[e] for e in self.EVENTS] for p in player_profiles])
#         self.name_dict = {p['id']: p['name'] for p in player_profiles}
#         self.prob_1b_to_home_on_double = 0.40
#         self.prob_2b_to_home_on_single = 0.60

#     # ---- 回傳每一場的得分向量 (向量化, 所有場次平行模擬) ----
#     def simulate_runs(self, lineup_indices, num_simulations, innings=9, seed=42,
#                       batch_size=2_000_000):
#         rng = np.random.default_rng(seed)
#         cum = np.cumsum(self.prob_matrix[np.asarray(lineup_indices)], axis=1)
#         out = np.empty(num_simulations, dtype=np.int32)
#         done = 0
#         while done < num_simulations:
#             g = min(batch_size, num_simulations - done)
#             out[done:done + g] = self._simulate_batch(g, cum, rng, innings)
#             done += g
#         return out

#     def _simulate_batch(self, G, cum, rng, innings):
#         total = np.zeros(G, dtype=np.int32)
#         batter = np.zeros(G, dtype=np.int64)
#         b1 = np.zeros(G, dtype=np.int8); b2 = np.zeros(G, dtype=np.int8); b3 = np.zeros(G, dtype=np.int8)
#         for _ in range(innings):
#             outs = np.zeros(G, dtype=np.int8)
#             b1[:] = 0; b2[:] = 0; b3[:] = 0
#             active = outs < 3
#             while active.any():
#                 idx = np.where(active)[0]
#                 u = rng.random(idx.shape[0])
#                 ev = (u[:, None] >= cum[batter[idx]]).sum(axis=1)

#                 m = ev == 0
#                 if m.any(): outs[idx[m]] += 1

#                 m = ev == 1                                   # BB
#                 if m.any():
#                     g = idx[m]; o1, o2, o3 = b1[g].copy(), b2[g].copy(), b3[g].copy()
#                     loaded = (o1 == 1) & (o2 == 1) & (o3 == 1)
#                     total[g[loaded]] += 1
#                     n1 = np.ones_like(o1)
#                     n2 = np.where(o1 == 1, 1, o2); n3 = np.where((o1 == 1) & (o2 == 1), 1, o3)
#                     n2 = np.where(loaded, 1, n2); n3 = np.where(loaded, 1, n3)
#                     b1[g], b2[g], b3[g] = n1, n2, n3

#                 m = ev == 2                                   # 1B
#                 if m.any():
#                     g = idx[m]; o1, o2, o3 = b1[g].copy(), b2[g].copy(), b3[g].copy()
#                     total[g] += o3
#                     has2 = o2 == 1
#                     score2 = has2 & (rng.random(g.shape[0]) < self.prob_2b_to_home_on_single)
#                     total[g[score2]] += 1
#                     b1[g] = 1; b2[g] = o1
#                     b3[g] = np.where(has2 & ~score2, 1, 0).astype(np.int8)

#                 m = ev == 3                                   # 2B
#                 if m.any():
#                     g = idx[m]; o1, o2, o3 = b1[g].copy(), b2[g].copy(), b3[g].copy()
#                     total[g] += (o2 + o3)
#                     has1 = o1 == 1
#                     score1 = has1 & (rng.random(g.shape[0]) < self.prob_1b_to_home_on_double)
#                     total[g[score1]] += 1
#                     b1[g] = 0; b2[g] = 1
#                     b3[g] = np.where(has1 & ~score1, 1, 0).astype(np.int8)

#                 m = ev == 4                                   # 3B
#                 if m.any():
#                     g = idx[m]; total[g] += (b1[g] + b2[g] + b3[g]); b1[g] = 0; b2[g] = 0; b3[g] = 1

#                 m = ev == 5                                   # HR
#                 if m.any():
#                     g = idx[m]; total[g] += (b1[g] + b2[g] + b3[g] + 1); b1[g] = 0; b2[g] = 0; b3[g] = 0

#                 batter[idx] = (batter[idx] + 1) % 9
#                 active = outs < 3
#         return total

#     def evaluate_lineup(self, lineup_indices, num_simulations=1000, seed=42, return_runs=False):
#         """回傳 (mean, game_std, sem). game_std = 單場得分標準差; sem = 平均數的標準誤。"""
#         runs = self.simulate_runs(lineup_indices, num_simulations, seed=seed)
#         mean = runs.mean()
#         game_std = runs.std(ddof=1)
#         sem = game_std / np.sqrt(num_simulations)
#         if return_runs:
#             return mean, game_std, sem, runs
#         return mean, game_std, sem


class BaseballSimulator:
    """
    新版向量化 Monte Carlo simulator。

    事件: ['SINGLE_OUT', 'DOUBLE_OUT', 'TRIPLE_OUT', 'BB', '1B', '2B', '3B', 'HR']

    出局規則與 BaseballModel 完全一致:
      * SINGLE_OUT : +1 出局, 跑者不動。
      * DOUBLE_OUT : 一壘有封殺跑者時 -> 清掉一壘跑者 + 打者出局 (+2 出局,
                     二三壘跑者留原位); 一壘無人時 -> 退化成 +1 出局。
      * TRIPLE_OUT : 直接結束半局 (出局數記滿 3)。

    向量化: 所有場次平行模擬, 可支援百萬 / 千萬場規模。
    evaluate_lineup 回傳 (平均得分, 單場得分標準差, 標準誤)。
    """
    EVENTS = ['SINGLE_OUT', 'DOUBLE_OUT', 'TRIPLE_OUT', 'BB', '1B', '2B', '3B', 'HR']

    def __init__(self, player_profiles):
        self.event_types = self.EVENTS
        self.player_profiles = player_profiles
        self.prob_matrix = np.array([[p[e] for e in self.EVENTS] for p in player_profiles])
        self.name_dict = {p['id']: p['name'] for p in player_profiles}
        self.prob_1b_to_home_on_double = 0.40
        self.prob_2b_to_home_on_single = 0.60

    def simulate_runs(self, lineup_indices, num_simulations, innings=9, seed=42,
                      batch_size=2_000_000):
        rng = np.random.default_rng(seed)
        cum = np.cumsum(self.prob_matrix[np.asarray(lineup_indices)], axis=1)
        out = np.empty(num_simulations, dtype=np.int32)
        done = 0
        while done < num_simulations:
            g = min(batch_size, num_simulations - done)
            out[done:done + g] = self._simulate_batch(g, cum, rng, innings)
            done += g
        return out

    def _simulate_batch(self, G, cum, rng, innings):
        total = np.zeros(G, dtype=np.int32)
        batter = np.zeros(G, dtype=np.int64)
        b1 = np.zeros(G, dtype=np.int8); b2 = np.zeros(G, dtype=np.int8); b3 = np.zeros(G, dtype=np.int8)
        for _ in range(innings):
            outs = np.zeros(G, dtype=np.int8)
            b1[:] = 0; b2[:] = 0; b3[:] = 0
            active = outs < 3
            while active.any():
                idx = np.where(active)[0]
                u = rng.random(idx.shape[0])
                ev = (u[:, None] >= cum[batter[idx]]).sum(axis=1)

                # --- SINGLE_OUT (id 0): +1 出局, 跑者不動 ---
                m = ev == 0
                if m.any():
                    outs[idx[m]] += 1

                # --- DOUBLE_OUT (id 1) ---
                m = ev == 1
                if m.any():
                    g = idx[m]
                    has1 = b1[g] == 1
                    g_dp = g[has1]
                    if g_dp.size:
                        b1[g_dp] = 0          # 清掉一壘封殺跑者 (二三壘不動)
                        outs[g_dp] += 2       # 打者也出局
                    g_sg = g[~has1]
                    if g_sg.size:
                        outs[g_sg] += 1       # 一壘無人 -> 退化成普通一出局

                # --- TRIPLE_OUT (id 2): 直接結束半局 ---
                m = ev == 2
                if m.any():
                    outs[idx[m]] = 3

                # --- BB (id 3): 強迫進壘 ---
                m = ev == 3
                if m.any():
                    g = idx[m]; o1, o2, o3 = b1[g].copy(), b2[g].copy(), b3[g].copy()
                    loaded = (o1 == 1) & (o2 == 1) & (o3 == 1)
                    total[g[loaded]] += 1
                    n1 = np.ones_like(o1)
                    n2 = np.where(o1 == 1, 1, o2); n3 = np.where((o1 == 1) & (o2 == 1), 1, o3)
                    n2 = np.where(loaded, 1, n2); n3 = np.where(loaded, 1, n3)
                    b1[g], b2[g], b3[g] = n1, n2, n3

                # --- 1B (id 4) ---
                m = ev == 4
                if m.any():
                    g = idx[m]; o1, o2, o3 = b1[g].copy(), b2[g].copy(), b3[g].copy()
                    total[g] += o3
                    has2 = o2 == 1
                    score2 = has2 & (rng.random(g.shape[0]) < self.prob_2b_to_home_on_single)
                    total[g[score2]] += 1
                    b1[g] = 1; b2[g] = o1
                    b3[g] = np.where(has2 & ~score2, 1, 0).astype(np.int8)

                # --- 2B (id 5) ---
                m = ev == 5
                if m.any():
                    g = idx[m]; o1, o2, o3 = b1[g].copy(), b2[g].copy(), b3[g].copy()
                    total[g] += (o2 + o3)
                    has1 = o1 == 1
                    score1 = has1 & (rng.random(g.shape[0]) < self.prob_1b_to_home_on_double)
                    total[g[score1]] += 1
                    b1[g] = 0; b2[g] = 1
                    b3[g] = np.where(has1 & ~score1, 1, 0).astype(np.int8)

                # --- 3B (id 6) ---
                m = ev == 6
                if m.any():
                    g = idx[m]; total[g] += (b1[g] + b2[g] + b3[g]); b1[g] = 0; b2[g] = 0; b3[g] = 1

                # --- HR (id 7) ---
                m = ev == 7
                if m.any():
                    g = idx[m]; total[g] += (b1[g] + b2[g] + b3[g] + 1); b1[g] = 0; b2[g] = 0; b3[g] = 0

                # 雙殺從兩出局狀態觸發會變 4 出局, 夾回 3。
                np.clip(outs, 0, 3, out=outs)

                batter[idx] = (batter[idx] + 1) % 9
                active = outs < 3
        return total

    def evaluate_lineup(self, lineup_indices, num_simulations=1000, seed=42, return_runs=False):
        """回傳 (mean, game_std, sem). game_std = 單場得分標準差; sem = 平均數的標準誤。"""
        runs = self.simulate_runs(lineup_indices, num_simulations, seed=seed)
        mean = runs.mean()
        game_std = runs.std(ddof=1)
        sem = game_std / np.sqrt(num_simulations)
        if return_runs:
            return mean, game_std, sem, runs
        return mean, game_std, sem


class BaseballModel:
    """
    Analytical baseball lineup evaluator following Bukiet, Harold & Palacios (1997),
    "A Markov Chain Approach to Baseball" (Operations Research 45(1), 14-23).
 
    Each plate appearance is a transition in a 25-state Markov chain
    (8 base configurations x 3 out-counts, plus one absorbing 3-out state).
    The lineup's nine batters apply their transition matrices in order, and -
    crucially, per Bukiet eq. (3) - the batting order CARRIES OVER across innings:
    the leadoff hitter of inning n+1 is whoever was due up when the third out of
    inning n was recorded, NOT a reset to the top of the order.
 
    This implementation also supports a granular out classification
    (single / double / triple outs), letting double plays clear a forced runner
    (Bukiet's nonzero C/E blocks) and triple plays end the half-inning outright.
    """
 
    def __init__(self, player_profiles):
        self.event_types = ['SINGLE_OUT', 'DOUBLE_OUT', 'TRIPLE_OUT',
                            'BB', '1B', '2B', '3B', 'HR']
        self.player_profiles = player_profiles
 
        self.prob_matrix = np.array(
            [[p[event] for event in self.event_types] for p in player_profiles]
        )
        self.name_dict = {p['id']: p['name'] for p in player_profiles}
 
        # Deterministic runner-advancement parameters (D'Esopo-Lefkowitz style,
        # with two stochastic extra-base rules).
        self.prob_1b_to_home_on_double = 0.40   # runner on 1st scores on a double
        self.prob_2b_to_home_on_single = 0.60   # runner on 2nd scores on a single
 
        # 25-state framework: 8 base configs x {0,1,2} outs + absorbing "3 outs".
        self.base_configurations = ["Empty", "1B", "2B", "3B",
                                    "1B_2B", "1B_3B", "2B_3B", "Full"]
        self.state_to_idx = {}
        state_counter = 0
        for outs in [0, 1, 2]:
            for base in self.base_configurations:
                self.state_to_idx[f"{outs}Outs_{base}"] = state_counter
                state_counter += 1
        self.state_to_idx["3Outs"] = 24
 
    # ------------------------------------------------------------------ #
    # Runner advancement                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _count_runners(base: str) -> int:
        """Number of runners currently on base for a given base-configuration label."""
        if base == "Empty":
            return 0
        if base == "Full":
            return 3
        return len(base.split("_"))
 
    def _advance_runners_analytically(self, current_base: str, hit_type: str) -> dict:
        """
        Map (base configuration, hit type) to a distribution over
        (new base configuration, runs scored), matching the simulator's
        baserunning rules. Outs are handled separately in _compile_lineup_matrices.
        """
        n_on = self._count_runners(current_base)
 
        if hit_type == "HR":
            # Everyone on base plus the batter scores.
            return {("Empty", n_on + 1): 1.0}
 
        if hit_type == "3B":
            # All existing runners score; batter ends on third.
            return {("3B", n_on): 1.0}
 
        if hit_type == "BB":
            # Walk: runners advance only when forced.
            rules = {
                "Empty": ("1B", 0), "1B": ("1B_2B", 0), "2B": ("1B_2B", 0),
                "3B": ("1B_3B", 0), "1B_2B": ("Full", 0), "1B_3B": ("Full", 0),
                "2B_3B": ("Full", 0), "Full": ("Full", 1),
            }
            return {rules[current_base]: 1.0}
 
        if hit_type == "1B":
            # Batter to first; runner on third scores; runner on first to second.
            # Runner on second scores with prob p_score, else stops at third.
            if current_base == "Empty":  return {("1B", 0): 1.0}
            if current_base == "1B":     return {("1B_2B", 0): 1.0}
            if current_base == "3B":     return {("1B", 1): 1.0}
            if current_base == "1B_3B":  return {("1B_2B", 1): 1.0}
 
            p = self.prob_2b_to_home_on_single
            if current_base == "2B":
                return {("1B", 1): p, ("1B_3B", 0): 1.0 - p}
            if current_base == "1B_2B":
                # runner-on-2nd scores -> [1st,2nd] occupied (=1B_2B); else -> bases loaded
                return {("1B_2B", 1): p, ("Full", 0): 1.0 - p}
            if current_base == "2B_3B":
                return {("1B", 2): p, ("1B_3B", 1): 1.0 - p}
            if current_base == "Full":
                return {("1B_2B", 2): p, ("Full", 1): 1.0 - p}
 
        if hit_type == "2B":
            # Batter to second; runners on 2nd/3rd score; runner on 1st scores
            # with prob p_score, else stops at third.
            p = self.prob_1b_to_home_on_double
            if current_base == "Empty":  return {("2B", 0): 1.0}
            if current_base == "2B":     return {("2B", 1): 1.0}
            if current_base == "3B":     return {("2B", 1): 1.0}
            if current_base == "2B_3B":  return {("2B", 2): 1.0}
 
            if current_base == "1B":
                return {("2B", 1): p, ("2B_3B", 0): 1.0 - p}
            if current_base == "1B_2B":
                return {("2B", 2): p, ("2B_3B", 1): 1.0 - p}
            if current_base == "1B_3B":
                return {("2B", 2): p, ("2B_3B", 1): 1.0 - p}
            if current_base == "Full":
                return {("2B", 3): p, ("2B_3B", 2): 1.0 - p}
 
        return {("Empty", 0): 1.0}
 
    # ------------------------------------------------------------------ #
    # Transition-matrix compilation                                      #
    # ------------------------------------------------------------------ #
    def _compile_lineup_matrices(self, lineup_indices) -> list:
        """Build the 25x25 transition matrix P and expected-run matrix R for each batter."""
        lineup_matrices = []
        for idx in lineup_indices:
            probs = dict(zip(self.event_types, self.prob_matrix[idx]))
            P = np.zeros((25, 25))
            R_expected = np.zeros((25, 25))
 
            for outs in [0, 1, 2]:
                for base in self.base_configurations:
                    from_idx = self.state_to_idx[f"{outs}Outs_{base}"]
 
                    # 1. Single out: +1 out, runners hold.
                    p_s_out = probs["SINGLE_OUT"]
                    if p_s_out > 0:
                        if outs == 2:
                            P[from_idx, 24] += p_s_out
                        else:
                            P[from_idx, self.state_to_idx[f"{outs + 1}Outs_{base}"]] += p_s_out
 
                    # 2. Double out: +2 outs, clearing the forced runner at first
                    #    (only possible when a force exists at first).
                    p_d_out = probs["DOUBLE_OUT"]
                    if p_d_out > 0:
                        has_force_at_1b = ("1B" in base) or (base == "Full")
                        if outs < 2 and has_force_at_1b:
                            next_outs = outs + 2
                            if next_outs >= 3:
                                P[from_idx, 24] += p_d_out
                            else:
                                dp_base_mapping = {"1B": "Empty", "1B_2B": "2B",
                                                   "1B_3B": "3B", "Full": "2B_3B"}
                                target_base = dp_base_mapping[base]
                                P[from_idx, self.state_to_idx[f"{next_outs}Outs_{target_base}"]] += p_d_out
                        else:
                            # No double play available -> degrade to a single out.
                            if outs == 2:
                                P[from_idx, 24] += p_d_out
                            else:
                                P[from_idx, self.state_to_idx[f"{outs + 1}Outs_{base}"]] += p_d_out
 
                    # 3. Triple out: half-inning ends immediately.
                    p_t_out = probs["TRIPLE_OUT"]
                    if p_t_out > 0:
                        P[from_idx, 24] += p_t_out
 
                    # 4. Safe events (hits and walks): outs unchanged, runners advance.
                    for hit_type in ["1B", "2B", "3B", "HR", "BB"]:
                        prob = probs[hit_type]
                        if prob == 0:
                            continue
                        branches = self._advance_runners_analytically(base, hit_type)
                        for (new_base, runs), weight in branches.items():
                            to_idx = self.state_to_idx[f"{outs}Outs_{new_base}"]
                            transition_prob = prob * weight
                            P[from_idx, to_idx] += transition_prob
                            R_expected[from_idx, to_idx] += transition_prob * runs
 
            P[24, 24] = 1.0
            lineup_matrices.append((P, R_expected))
        return lineup_matrices
 
    # ------------------------------------------------------------------ #
    # Lineup evaluation                                                  #
    # ------------------------------------------------------------------ #
    def evaluate_lineup(self, lineup_indices, innings: int = 9) -> float:
        """
        Expected runs for a full game, following Bukiet eq. (3).
 
        The batting order carries across innings: when the third out is recorded,
        play resumes in the next inning with a clean (0 outs, bases empty) state
        but with the NEXT batter due up - not a reset to the leadoff hitter.
 
        We propagate a probability distribution over (inning, batter-due, game-state)
        plate appearance by plate appearance, peeling off the mass that reaches the
        3-out absorbing state into the following inning's leadoff state, and
        accumulating expected runs along the way. This is exact (zero variance).
        """
        mats = self._compile_lineup_matrices(lineup_indices)
 
        # dist[(inning, batter_due)] -> 25-vector of live probability mass.
        dist = {(0, 0): np.zeros(25)}
        dist[(0, 0)][0] = 1.0
        total_expected_runs = 0.0
 
        # Each pass advances every live (inning, batter) cohort by one plate appearance.
        # The loop terminates once all mass has completed the final inning.
        while dist:
            next_dist = {}
            for (inning, batter), u in dist.items():
                if u.sum() < 1e-15:
                    continue
                P_b, R_b = mats[batter]
 
                total_expected_runs += np.sum(u[:, None] * R_b)
 
                u_next = u @ P_b
                absorbed = u_next[24]          # mass that just made the 3rd out
                live = u_next.copy()
                live[24] = 0.0                 # remaining mass stays in this inning
                next_batter = (batter + 1) % 9
 
                if live.sum() > 1e-16:
                    key = (inning, next_batter)
                    if key in next_dist:
                        next_dist[key] += live
                    else:
                        next_dist[key] = live
 
                # Carry absorbed mass to the next inning's leadoff state,
                # keeping the same (next) batter. Drop it after the final inning.
                if absorbed > 1e-16 and inning + 1 < innings:
                    v = np.zeros(25)
                    v[0] = absorbed
                    key = (inning + 1, next_batter)
                    if key in next_dist:
                        next_dist[key] += v
                    else:
                        next_dist[key] = v
 
            dist = next_dist
 
        return total_expected_runs