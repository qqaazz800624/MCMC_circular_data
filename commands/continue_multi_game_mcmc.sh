#!/bin/bash

declare -A initial_x_dict

initial_x_dict["random_insertion_proposal"]="3,1,4,2,0,8,7,5,6"
initial_x_dict["directional_reversal_proposal"]="2,1,4,0,3,8,6,5,7"   
initial_x_dict["k_cycle_shift_proposal"]="2,0,1,4,3,5,6,7,8"           


export EXPERIMENT_NAME="2nd_1000steps"
export TEAM="LAD"
export YEAR="2024"

echo "Starting batch experiments for ${#initial_x_dict[@]} proposals..."

for PROPOSAL in "${!initial_x_dict[@]}"
do
    INITIAL_X="${initial_x_dict[$PROPOSAL]}"
    
    echo "============================================================"
    echo "Currently Running: $PROPOSAL"
    echo "Starting from initial_x: $INITIAL_X"
    echo "============================================================"
    
    python scripts/game_mcmc_simulator.py \
        --num_initials 1 \
        --num_sims_per_step 100000 \
        --max_steps 1000 \
        --proposal "$PROPOSAL" \
        --tau 5 \
        --data_dir "results" \
        --initial_x "$INITIAL_X" \
        --experiment_name "$EXPERIMENT_NAME" \
        --lineup_filename "player_profiles_${TEAM}_${YEAR}.json"
        
    echo "Finished: $PROPOSAL"
    echo ""
done

echo "All 2nd stage experiments completed successfully!"