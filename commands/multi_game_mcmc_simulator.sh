#!/bin/bash

PROPOSALS=(
    "random_swap_proposal"
    "random_insertion_proposal"
)

export team="LAD"
export year="2024"

echo "Starting batch experiments for ${#PROPOSALS[@]} proposals..."

for PROPOSAL in "${PROPOSALS[@]}"
do
    echo "============================================================"
    echo "Currently Running: $PROPOSAL"
    echo "============================================================"
    
    python scripts/game_mcmc_simulator.py \
        --num_initials 1 \
        --num_sims_per_step 1000 \
        --max_steps 5000 \
        --proposal "$PROPOSAL" \
        --tau 5 \
        --data_dir "results" \
        --lineup_filename "player_profiles_${team}_${year}.json"
        
    echo "Finished: $PROPOSAL"
    echo ""
done

echo "All experiments completed successfully!"