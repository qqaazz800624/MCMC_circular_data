#!/bin/bash

export PROPOSAL="block_pair_exchange_proposal"

export TEAM="LAD"
export YEAR="2024"
export TAU=0.005
export MAX_STEPS=3000
export INITIAL_SEED=49
export SEED=43
export EXPERIMENT_NAME="INITIAL_SEED_${INITIAL_SEED}_${MAX_STEPS}steps"

echo "Starting mcmc experiments for ${PROPOSAL} proposal..."

echo "============================================================"
echo "Currently Running: $PROPOSAL"
echo "Experiment Name: $EXPERIMENT_NAME"
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
        --seed "$SEED" 
        
echo "Started: $PROPOSAL mcmc simulation ..."
echo ""

wait

echo "Experiment of $EXPERIMENT_NAME completed successfully!"