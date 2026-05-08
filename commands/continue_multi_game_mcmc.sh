#!/bin/bash

declare -A initial_x_dict

initial_x_dict["random_swap_proposal"]="2,4,1,0,7,3,6,5,8"
initial_x_dict["random_insertion_proposal"]="1,0,4,2,3,7,5,6,8"   
initial_x_dict["directional_reversal_proposal"]="1,4,0,2,3,8,7,5,6"           


export EXPERIMENT_NAME="3rd_1000steps"
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

echo "All 3rd stage experiments completed successfully!"