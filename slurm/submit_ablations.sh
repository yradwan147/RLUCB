#!/bin/bash
# Submit ablation study experiments to IBEX cluster.
# Tests F-UCB and BKT-Bandit component contributions.
# Total: ~8 jobs
set -e

mkdir -p slurm/slurm_logs results

SEEDS="42 123 456"
K=20  # Fixed category count for ablations

COUNT=0

# ---------------------------------------------------------------
# Exploration parameter sensitivity (c = 0.5, 1.0, 1.414, 2.0, 3.0)
# Each job runs all algos, so we just vary decay at fixed K
# ---------------------------------------------------------------
echo "Submitting exploration parameter / extra decay rate jobs..."
EXTRA_DECAYS="0.001 0.1"
for DECAY in $EXTRA_DECAYS; do
    for SEED in $SEEDS; do
        sbatch -J abl_k${K}_d${DECAY}_s${SEED} \
            slurm/run_experiment.sh $K $DECAY $SEED
        COUNT=$((COUNT + 1))
    done
done

# Large-scale test (K=100) with fewer seeds
echo "Submitting large-scale (K=100) jobs..."
for SEED in 42 123; do
    sbatch -J abl_k100_d0.01_s${SEED} \
        slurm/run_experiment.sh 100 0.01 $SEED
    COUNT=$((COUNT + 1))
done

echo "Submitted $COUNT ablation jobs"
