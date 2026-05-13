#!/bin/bash

export PROPOSAL="random_insertion_proposal"
export EXPERIMENT_NAME="1st_1000steps"
export team="BOS"
export year="2024"
export SEED=42

echo "Running MCMC simulator for $EXPERIMENT_NAME with $PROPOSAL and 1000 steps..."

python scripts/game_mcmc_simulator.py \
    --num_initials 1 \
    --num_sims_per_step 100000 \
    --max_steps 1000 \
    --proposal "$PROPOSAL" \
    --tau 5 \
    --data_dir "results" \
    --experiment_name "$EXPERIMENT_NAME" \
    --lineup_filename "player_profiles_${team}_${year}.json" \
    --seed $SEED

echo "Experiment completed successfully!"