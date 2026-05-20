#!/bin/bash

PROPOSALS=(
    "random_swap_proposal"
    "random_insertion_proposal"
    "directional_reversal_proposal"
    "k_cycle_shift_proposal"
    "block_pair_exchange_proposal"
)

export TEAM="LAD"
export YEAR="2024"
export TAU=20
export SEED=42
export EXPERIMENT_NAME="tau_${TAU}_1st_1000steps"

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
        --tau "$TAU" \
        --data_dir "results" \
        --experiment_name "$EXPERIMENT_NAME" \
        --lineup_filename "player_profiles_${TEAM}_${YEAR}.json" \
        --seed "$SEED" &
        
    echo "Started: $PROPOSAL in the background."
    echo ""
done

wait

echo "All experiments completed successfully!"