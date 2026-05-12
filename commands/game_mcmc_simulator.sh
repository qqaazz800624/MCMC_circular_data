#!/bin/bash

export PROPOSAL="block_pair_exchange_proposal"
export EXPERIMENT_NAME="1st_1000steps"
export team="LAA"
export year="2023"

echo "Running MCMC simulator for $EXPERIMENT_NAME with $PROPOSAL and 1000 steps..."

python scripts/game_mcmc_simulator.py \
    --num_initials 1 \
    --num_sims_per_step 100000 \
    --max_steps 1000 \
    --proposal "$PROPOSAL" \
    --tau 5 \
    --data_dir "results" \
    --experiment_name "$EXPERIMENT_NAME" \
    --lineup_filename "player_profiles_${team}_${year}.json"

echo "Experiment completed successfully!"