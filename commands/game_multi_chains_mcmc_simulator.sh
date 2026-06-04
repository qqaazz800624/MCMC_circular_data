#!/bin/bash

export PROPOSAL="random_swap_proposal"
export NUM_CHAINS=30

export TEAM="LAD"
export YEAR="2024"
export TAU=0.005
export MAX_STEPS=3000
export INITIAL_SEED=73
export SEED=$((INITIAL_SEED + 1000))
export EXPERIMENT_NAME="INITIAL_SEED_${INITIAL_SEED}_${MAX_STEPS}steps_${NUM_CHAINS}chains"

echo "============================================================"
echo "Starting experiment: $PROPOSAL with $NUM_CHAINS chains"
echo "============================================================"

python scripts/game_multi_chains_mcmc_simulator.py \
    --num_initials "$NUM_CHAINS" \
    --num_sims_per_step 100000 \
    --max_steps "$MAX_STEPS" \
    --proposal "$PROPOSAL" \
    --tau "$TAU" \
    --data_dir "results" \
    --cache_dir "/data/share/mcmc" \
    --experiment_name "$EXPERIMENT_NAME" \
    --lineup_filename "player_profiles_${TEAM}_${YEAR}.json" \
    --initial_seed "$INITIAL_SEED" \
    --seed "$SEED"

wait

echo "Experiment ${EXPERIMENT_NAME} completed successfully!"