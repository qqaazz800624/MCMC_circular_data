#!/bin/bash

python scripts/analyze_landscape.py \
    --n 9 \
    --g 4.9 4.8 4.7 4.6 4.5 4.4 4.3 4.2 4.1\
    --alpha 1.0 \
    --beta 10 \
    --tau 5 \
    --save_dir "results"