#!/bin/bash

python scripts/run_experiment.py \
    --num_experiments 1 \
    --max_steps 1000 \
    --n 9 \
    --g 4.9 4.8 4.7 4.6 4.5 4.4 4.3 4.2 4.1 \
    --x 0.8 0.775 0.75 0.725 0.7 0.675 0.65 0.625 0.6 \
    --objective "toy_objective_1" \
    --proposal "directional_reversal_proposal" \
    --alpha 1.0 \
    --beta 100 \
    --tau 5 \
    --true_max_F 8.55078 \
    --save_dir "results"