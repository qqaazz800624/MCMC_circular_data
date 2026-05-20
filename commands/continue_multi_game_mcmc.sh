#!/bin/bash

declare -A initial_x_dict

initial_x_dict["random_swap_proposal"]="2,1,4,0,3,7,5,8,6"
initial_x_dict["random_insertion_proposal"]="2,1,4,0,3,7,5,8,6"   
initial_x_dict["directional_reversal_proposal"]="4,2,1,0,3,8,5,7,6"
initial_x_dict["k_cycle_shift_proposal"]="2,1,4,0,3,7,5,6,8"
initial_x_dict["block_pair_exchange_proposal"]="2,4,1,0,3,5,7,6,8"           

export TAU=0.01
export EXPERIMENT_NAME="tau_${TAU}_3rd_1000steps"
export SEED=42
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
        --tau "$TAU" \
        --data_dir "results" \
        --initial_x "$INITIAL_X" \
        --experiment_name "$EXPERIMENT_NAME" \
        --seed "$SEED" \
        --lineup_filename "player_profiles_${TEAM}_${YEAR}.json" &
        
    echo "Started: $PROPOSAL experiment in the background."
    echo ""
done

wait

echo "All experiments completed successfully!"