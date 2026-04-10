#!/bin/bash

python scripts/run_simulation.py \
    --steps 1000 \
    --n 9 \
    --objective "toy_objective_1" \
    --proposal "random_swap_proposal" \
    --alpha 1.0 \
    --beta 0.005 \
    --tau 0.5 \
    --seed 42 