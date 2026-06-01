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
export TAU=0.005
export MAX_STEPS=3000
export INITIAL_SEED=71
export SEED=$((INITIAL_SEED + 1000))
export EXPERIMENT_NAME="INITIAL_SEED_${INITIAL_SEED}_${MAX_STEPS}steps"

echo "Starting batch experiments for ${#PROPOSALS[@]} proposals..."

for PROPOSAL in "${PROPOSALS[@]}"
do
    echo "============================================================"
    echo "Currently Running: $PROPOSAL"
    echo "============================================================"
    
    python scripts/game_mcmc_simulator.py \
        --num_initials 1 \
        --num_sims_per_step 100000 \
        --max_steps "$MAX_STEPS" \
        --proposal "$PROPOSAL" \
        --tau "$TAU" \
        --data_dir "results" \
        --cache_dir "/data/share/mcmc" \
        --experiment_name "$EXPERIMENT_NAME" \
        --lineup_filename "player_profiles_${TEAM}_${YEAR}.json" \
        --initial_seed "$INITIAL_SEED" \
        --seed "$SEED" &
        
    echo "Started: $PROPOSAL in the background."
    echo ""
done

wait

echo "All experiments of ${EXPERIMENT_NAME} completed successfully!"