#!/bin/bash

if [ -z "$1" ]; then
    echo " Please provide Job ID！"
    exit 1
fi

export job_id=$1

echo "============================================================"
echo "Monitoring log file for job ID: $job_id"
echo "Hint: Press Ctrl + C at any time to exit viewing, it will not affect the program execution."
echo "============================================================"

tail -f results/toy_experiment_${job_id}.log