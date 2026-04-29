#!/bin/bash

python scripts/game_mcmc_simulator.py \
    --num_initials 1 \
    --num_sims_per_step 1000 \
    --max_steps 5000 \
    --proposal "random_swap_proposal" \
    --tau 5 \
    --data_dir "results" \
    --lineup_filename "player_profiles_LAD_2024.json"