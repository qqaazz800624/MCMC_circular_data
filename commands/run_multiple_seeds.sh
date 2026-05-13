#!/bin/bash
# run_multiple_seeds.sh

EXPERIMENT_NAME="random_initials_test"
PROPOSAL="random_swap_proposal"

echo "Start 5 seeds..."

for SEED in 42 43 44 45 46
do
    echo "Starting background Seed: $SEED"
    python scripts/game_mcmc_simulator.py \
        --num_initials 1 \
        --num_sims_per_step 100000 \
        --max_steps 1000 \
        --proposal "$PROPOSAL" \
        --tau 5 \
        --data_dir "results" \
        --experiment_name "${EXPERIMENT_NAME}_seed${SEED}" \
        --seed $SEED \
        --lineup_filename "player_profiles_LAD_2024.json" &
done

wait
echo "All 5 seeds experiments completed!"