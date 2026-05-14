#!/bin/bash

PROPOSALS=(
    "random_swap_proposal"
    "random_insertion_proposal"
    "directional_reversal_proposal"
    "k_cycle_shift_proposal"
    "block_pair_exchange_proposal"
)

export team="NYY"
export year="2024"
export tau=5
export EXPERIMENT_NAME="tau_${tau}_1000steps"

echo "Starting batch experiments for ${#PROPOSALS[@]} proposals..."

for PROPOSAL in "${PROPOSALS[@]}"
do
    echo "============================================================"
    echo "Currently Running: $PROPOSAL"
    echo "============================================================"
    
    python scripts/game_mcmc_simulator.py \
        --num_initials 1 \
        --num_sims_per_step 100000 \
        --max_steps 1000 \
        --proposal "$PROPOSAL" \
        --tau "$tau" \
        --data_dir "results" \
        --experiment_name "$EXPERIMENT_NAME" \
        --lineup_filename "player_profiles_${team}_${year}.json" &
        
    echo "Started: $PROPOSAL in the background."
    echo ""
done

wait

echo "All experiments completed successfully!"