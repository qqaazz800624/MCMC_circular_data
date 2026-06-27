#!/bin/bash

if [ -z "$1" ]; then
    echo " Please provide Job ID!"
    exit 1
fi

export job_id=$1

echo "============================================================"
echo "Monitoring all log files for job ID: $job_id"
echo "Hint: Press Ctrl + C at any time to exit viewing, it will not affect the program execution."
echo "============================================================"

# 利用萬用字元監控該 Job ID 底下的所有 log 檔
# 這會包含 toy_experiment_main_xxx.log 以及所有 proposal_xxx.log
tail -f results/*_${job_id}.log