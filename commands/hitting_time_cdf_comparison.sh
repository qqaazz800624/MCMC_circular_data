#!/bin/bash

python scripts/hitting_time_cdf_comparison.py \
    --data_dir "/home/qqaazz800624/MCMC_circular_data/results" \
    --save_dir "results" \
    --tau 5.0 \
    --proposals "k_cycle_shift_proposal" \
                "hybrid_reversal_kcycle_proposal" \
                "hybrid_swap_kcycle_proposal" \
                "hybrid_insertion_kcycle_proposal" \
                "hybrid_kcycle_block_exchange_proposal"