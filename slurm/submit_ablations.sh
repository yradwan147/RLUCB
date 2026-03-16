#!/bin/bash
# Submit ablation study experiments to IBEX cluster.
# Tests F-UCB and BKT-Bandit component contributions.
set -e

mkdir -p slurm/slurm_logs results

SEEDS="42 123 456 789 1024"
K=20  # Fixed category count for ablations
DECAY=0.01  # Fixed decay rate

COUNT=0

# ---------------------------------------------------------------
# Group 1: F-UCB ablations (vary urgency weight and decay awareness)
# ---------------------------------------------------------------
echo "Submitting F-UCB ablation jobs..."
for SEED in $SEEDS; do
    # Full F-UCB (baseline for ablation)
    sbatch -J abl_fucb_full_s${SEED} \
        slurm/run_experiment.sh fucb $K $DECAY $SEED
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# Group 2: Exploration parameter sensitivity
# ---------------------------------------------------------------
echo "Submitting exploration parameter sensitivity jobs..."
EXPLORATION_PARAMS="0.5 1.0 1.414 2.0 3.0"
for C in $EXPLORATION_PARAMS; do
    for SEED in $SEEDS; do
        sbatch -J abl_c${C}_s${SEED} \
            slurm/run_experiment.sh fucb $K $DECAY $SEED 100 10000
        COUNT=$((COUNT + 1))
    done
done

echo "Submitted $COUNT ablation jobs"
