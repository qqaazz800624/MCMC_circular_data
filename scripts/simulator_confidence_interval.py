import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from objectives import BaseballSimulator, BaseballModel
import json

team = "BOS"
year = 2024

json_path = f'/home/qqaazz800624/MCMC_circular_data/results/player_profiles_{team}_{year}.json'

with open(json_path, 'r') as f:
    player_profiles = json.load(f)


analytic_model = BaseballModel(player_profiles)
test_lineup = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
ANALYTIC_EXPECTATION = analytic_model.evaluate_lineup(test_lineup, innings=9)


def summarize(runs, num_sims):
    mean = runs.mean()
    game_std = runs.std(ddof=1)          # 單場得分標準差
    sem = game_std / np.sqrt(num_sims)   # 平均數的標準誤
    lo, hi = mean - 1.96 * sem, mean + 1.96 * sem
    return mean, game_std, sem, lo, hi


def main():
    print(f"Initiate (Benchmark) for team {team} in {year}...")
    random_seed = 43
    simulator = BaseballSimulator(player_profiles)

    #test_lineup = test_lineup

    # ------------------------------------------------------------------ #
    # 1. Original benchmark: 100,000 games + timing + 1000 MCMC steps estimation #
    # ------------------------------------------------------------------ #
    num_sims = 100000
    start_time = time.time()
    mean, game_std, sem, runs = simulator.evaluate_lineup(
        test_lineup, num_simulations=num_sims, seed=random_seed, return_runs=True
    )
    elapsed_time = time.time() - start_time

    print("-" * 48)
    print(f"Simulation : {num_sims} games")
    print(f"Total time: {elapsed_time:.4f} seconds")
    print(f"Expected score: {mean:.3f} runs/game")
    print(f"Per-game std dev: {game_std:.3f} runs/game")
    print(f"Std error of mean (SEM): {sem:.4f} runs/game")
    print(f"95% CI of mean: [{mean - 1.96*sem:.4f}, {mean + 1.96*sem:.4f}]")
    print(f"Estimated 1000 MCMC steps time: {(elapsed_time * 1000) / 60:.2f} minutes")
    print("-" * 48)

    # ------------------------------------------------------------------ #
    # 2. Convergence Table: mean / per-game std / SEM / 95% CI           #
    # ------------------------------------------------------------------ #
    print("\n Convergence Table (As sample size increases, 95% CI should gradually tighten and encompass the analytic true value)")
    print(f"{'N games':>12} | {'mean':>8} | {'game_std':>8} | {'SEM':>8} | {'95% CI':>22} | in CI")
    print("-" * 80)
    sample_sizes = [10000, 100000, 1000000, 10000000]
    for N in sample_sizes:
        r = simulator.simulate_runs(test_lineup, num_simulations=N, seed=random_seed)
        m, gs, se, lo, hi = summarize(r, N)
        inside = "" if ANALYTIC_EXPECTATION is None else ("Yes" if lo <= ANALYTIC_EXPECTATION <= hi else "No")
        print(f"{N:>12,} | {m:>8.4f} | {gs:>8.4f} | {se:>8.5f} | [{lo:>8.4f}, {hi:>8.4f}] | {inside}")

    # 留下最大樣本的逐場結果供畫圖
    big_runs = r

    # ------------------------------------------------------------------ #
    # 3. Plot: (Left) Cumulative mean + 95% CI convergence  (Right) Per-game score distribution #
    # ------------------------------------------------------------------ #
    Ns = np.unique(np.round(np.logspace(3, np.log10(big_runs.size), 40)).astype(int))
    csum = np.cumsum(big_runs.astype(np.float64))
    csq = np.cumsum(big_runs.astype(np.float64) ** 2)
    means, los, his = [], [], []
    for n in Ns:
        mu = csum[n - 1] / n
        var = (csq[n - 1] / n - mu ** 2) * n / (n - 1)
        se = np.sqrt(var) / np.sqrt(n)
        means.append(mu); los.append(mu - 1.96 * se); his.append(mu + 1.96 * se)
    means, los, his = map(np.array, (means, los, his))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.fill_between(Ns, los, his, alpha=0.25, color="#4C72B0", label="95% confidence interval")
    ax1.plot(Ns, means, color="#4C72B0", lw=1.8, label="Simulated cumulative mean")
    if ANALYTIC_EXPECTATION is not None:
        ax1.axhline(ANALYTIC_EXPECTATION, color="#C44E52", ls="--", lw=1.8,
                    label=f"Analytic expectation = {ANALYTIC_EXPECTATION:.4f}")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of simulated games  (log scale)")
    ax1.set_ylabel("Expected runs per game")
    ax1.set_title("Convergence of Monte Carlo mean")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    maxr = int(np.percentile(big_runs, 99.9))
    ax2.hist(big_runs, bins=np.arange(0, maxr + 2) - 0.5, density=True,
             color="#55A868", edgecolor="white", alpha=0.85)
    ax2.axvline(big_runs.mean(), color="#C44E52", ls="--", lw=1.8,
                label=f"Mean = {big_runs.mean():.4f}")
    ax2.set_xlabel("Runs scored in a 9-inning game")
    ax2.set_ylabel("Probability")
    ax2.set_title(f"Per-game run distribution (N = {big_runs.size:,})")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/benchmark_convergence_{team}_{year}.png", dpi=150, bbox_inches="tight")
    print(f"\n Plot saved: results/benchmark_convergence_{team}_{year}.png")


if __name__ == "__main__":
    main()