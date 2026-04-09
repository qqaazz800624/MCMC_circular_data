import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 定義目標函數 F(x) 與圓環拓撲
# ==========================================
def calculate_F(x, g, alpha=1.0, beta=0.01):
    """
    計算 Toy Example 的目標函數 F(x)
    包含絕對位置的線性項 (Linear term) 與圓環序列的連乘項 (Interaction term)
    """
    n = len(x)
    linear_term = np.sum(g * x)
    
    interaction_term = 0
    for i in range(n):
        prod = 1
        for j in range(5):
            prod *= x[(i + j) % n]
        interaction_term += prod
        
    return alpha * linear_term + beta * interaction_term

# ==========================================
# 2. 定義對稱提案分佈 J(x'|x) (Random Swap)
# ==========================================
def random_swap_proposal(x):
    """
    隨機挑選兩個相異位置並交換元素 (Symmetric Proposal)
    """
    n = len(x)
    x_prime = x.copy()
    i, j = np.random.choice(n, size=2, replace=False)
    x_prime[i], x_prime[j] = x_prime[j], x_prime[i]
    return x_prime

# ==========================================
# 3. 主演算法：Metropolis-Hastings 
# ==========================================
def run_mcmc(n_iterations=5000, n=9, alpha=1.0, beta=0.05, tau=50.0):
    """
    執行 MCMC 尋找排列空間 S_n 中的最佳解
    """
    # 初始化：隨機生成一個 1 到 n 的排列
    current_x = np.arange(1, n + 1)
    np.random.shuffle(current_x)
    
    # 生成嚴格遞減的權重 g (例如 n=9 時，g = [9, 8, 7, 6, 5, 4, 3, 2, 1])
    g = np.arange(n, 0, -1)
    
    # 計算初始分數
    current_F = calculate_F(current_x, g, alpha, beta)
    
    # 追蹤全域最佳解
    best_x = current_x.copy()
    best_F = current_F
    
    # 紀錄收斂軌跡
    history_F = [current_F]
    acceptance_count = 0
    
    # 開始 MCMC 迭代
    for t in range(n_iterations):
        # Step 1: 從 J(x'|x) 產生候選狀態
        proposed_x = random_swap_proposal(current_x)
        
        # Step 2: 計算目標函數差異 (Delta F)
        proposed_F = calculate_F(proposed_x, g, alpha, beta)
        delta_F = proposed_F - current_F
        
        # Step 3: 計算 MH 接受機率 (利用 Boltzmann 分佈，Z 互相抵消)
        # 為了避免 delta_F 很大時 np.exp 爆掉，分開判斷：
        if delta_F > 0:
            # 提案比現在好，絕對接受
            acceptance_prob = 1.0 
        else:
            # 提案比較差，根據溫度 tau 決定容忍度
            acceptance_prob = np.exp(delta_F / tau)
            
        # Step 4: 擲骰子決定是否接受
        if np.random.rand() <= acceptance_prob:
            current_x = proposed_x
            current_F = proposed_F
            acceptance_count += 1
            
            # 更新全域最佳解
            if current_F > best_F:
                best_x = current_x.copy()
                best_F = current_F
                
        # 紀錄歷史軌跡
        history_F.append(current_F)
        
    acceptance_rate = acceptance_count / n_iterations
    return best_x, best_F, history_F, acceptance_rate

# ==========================================
# 4. 執行與視覺化
# ==========================================
if __name__ == "__main__":
    np.random.seed(42) # 固定亂數種子方便重現結果
    
    N_ITER = 10000
    print(f"啟動 MCMC，執行 {N_ITER} 次迭代...")
    
    # 這裡的 beta 調得很小 (0.05)，因為連乘項最大可達 15120，如果不調小會完全吃掉線性項
    best_x, best_F, history_F, accept_rate = run_mcmc(
        n_iterations=N_ITER, 
        n=9, 
        alpha=1.0, 
        beta=0.05, 
        tau=20.0
    )
    
    print("-" * 30)
    print(f"整體接受率 (Acceptance Rate): {accept_rate:.2%}")
    print(f"找到的最佳狀態 (Best x): {best_x}")
    print(f"最高分數 (Best F): {best_F:.2f}")
    
    # 畫出收斂軌跡圖 (Trace Plot)
    plt.figure(figsize=(10, 5))
    plt.plot(history_F, color='royalblue', alpha=0.7)
    plt.axhline(y=best_F, color='red', linestyle='--', label='Global Best')
    plt.title('MCMC Trace Plot (Objective Function $F(x)$)')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()