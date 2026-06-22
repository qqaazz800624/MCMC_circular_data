#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=defq
#SBATCH --gres=gpu:pro6000:1
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data:/data
#SBATCH --container-image='/data/container-images/enroot/nvidia+pytorch+25.03-py3.sqsh'
#SBATCH --container-mount-home
#SBATCH --output=results/mcmc_run_%j.log


export PROPOSAL="von_mises_fisher_circular_insertion_proposal"
export NUM_CHAINS=200
export TEAM="LAD"
export YEAR="2024"
export TAU=0.005
export MAX_STEPS=3000
export INITIAL_SEED=10
export SEED=$((INITIAL_SEED + 1108))
export EXPERIMENT_NAME="INITIAL_SEED_${INITIAL_SEED}_${PROPOSAL}_${NUM_CHAINS}chains"

echo "Starting MCMC Experiment: $PROPOSAL"

cd $SLURM_SUBMIT_DIR

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

echo "Experiment $EXPERIMENT_NAME completed!"