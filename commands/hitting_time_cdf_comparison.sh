#!/bin/bash

python scripts/hitting_time_cdf_comparison.py \
    --data_dir "/home/qqaazz800624/MCMC_circular_data/results" \
    --save_dir "results" \
    --tau 5.0 \
    --proposals "random_swap_proposal" "hybrid_swap_reversal_proposal" "hybrid_swap_insertion_proposal" "hybrid_swap_block_exchange_proposal" "hybrid_swap_kcycle_proposal"