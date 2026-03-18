#!/bin/bash
# Submit experiments with new Whittle-based algorithms.
# Each job runs ALL 13 algorithms for one (K, decay, seed) config.
#
# Grid: 4 K × 3 decay × 3 seeds = 36 jobs
set -e

mkdir -p slurm/slurm_logs results

SEEDS="42 123 456"
CATEGORIES="6 20 50 100"
DECAY_RATES="0.005 0.01 0.05"

COUNT=0

for K in $CATEGORIES; do
    for DECAY in $DECAY_RATES; do
        for SEED in $SEEDS; do
            sbatch -J w_k${K}_d${DECAY}_s${SEED} \
                slurm/run_experiment.sh $K $DECAY $SEED
            COUNT=$((COUNT + 1))
        done
    done
done

echo "Submitted $COUNT jobs (each runs all 13 algorithms including Whittle variants)"
echo "Monitor: squeue -u \$USER"
echo "Cancel: squeue -u \$USER -o '%i %j' | grep 'w_k' | awk '{print \$1}' | xargs -r scancel"
