#!/bin/bash

export PROPOSAL="k_cycle_shift_proposal"
export EXPERIMENT_NAME="3rd_1000steps"

echo "Running MCMC simulator for $EXPERIMENT_NAME with $PROPOSAL and 1000 steps..."

python scripts/game_mcmc_simulator.py \
    --num_initials 1 \
    --num_sims_per_step 100000 \
    --max_steps 1000 \
    --proposal "$PROPOSAL" \
    --tau 5 \
    --data_dir "results" \
    --initial_x "2,0,1,4,3,5,6,7,8" \
    --experiment_name "$EXPERIMENT_NAME" \
    --lineup_filename "player_profiles_LAD_2024.json"

echo "Experiment completed successfully!"