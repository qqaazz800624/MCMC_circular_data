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
#SBATCH --output=results/toy_experiment_%j.log

# ==========================================
# Experiment Configuration
# ==========================================
PROPOSALS=(
    "random_swap_proposal"
    "random_insertion_proposal"
    "directional_reversal_proposal"
    "k_cycle_shift_proposal"
    "block_pair_exchange_proposal"
    "local_biased_circular_insertion"
    "von_mises_fisher_circular_insertion"
)

export TAU=20
export NUM_EXPERIMENTS=10000
export MAX_STEPS=50000
export INITIAL_SEED=42
export SAVE_DIR="results"

# Ensure the results directory exists for logs and outputs
mkdir -p "$SAVE_DIR"

echo "Starting batch experiments for ${#PROPOSALS[@]} proposals using SLURM..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"

cd $SLURM_SUBMIT_DIR

# ==========================================
# Run Experiments Sequentially
# ==========================================
for PROPOSAL in "${PROPOSALS[@]}"
do
    echo "============================================================"
    echo "Currently Running: $PROPOSAL"
    echo "============================================================"
    
    python scripts/run_experiment.py \
        --num_experiments "$NUM_EXPERIMENTS" \
        --max_steps "$MAX_STEPS" \
        --n 9 \
        --g 4.9 4.8 4.7 4.6 4.5 4.4 4.3 4.2 4.1 \
        --x 0.8 0.775 0.75 0.725 0.7 0.675 0.65 0.625 0.6 \
        --objective "toy_objective_1" \
        --proposal "$PROPOSAL" \
        --alpha 1.0 \
        --beta 100 \
        --tau "$TAU" \
        --true_max_F 8.55078 \
        --save_dir "$SAVE_DIR" \
        --seed "$INITIAL_SEED"
        
    echo "Finished: $PROPOSAL"
    echo ""
done

echo "All experiments completed successfully!"