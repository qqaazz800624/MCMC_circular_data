#!/bin/bash

python scripts/hitting_time_cdf_comparison.py \
    --data_dir "/home/qqaazz800624/MCMC_circular_data/results" \
    --save_dir "results" \
    --tau 5.0 \
    --baseline_proposal "directional_reversal_proposal" \
    --proposals "directional_reversal_proposal" \
                "hybrid_swap_reversal_proposal" \
                "hybrid_insertion_reversal_proposal" \
                "hybrid_reversal_block_exchange_proposal" \
                "hybrid_reversal_kcycle_proposal"