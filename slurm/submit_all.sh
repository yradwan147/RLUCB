#!/bin/bash
# Submit synthetic sweep experiments to IBEX cluster.
# Each job runs ALL 10 algorithms for one (K, decay, seed) config.
#
# Grid: 4 category counts × 3 decay rates × 3 seeds = 36 jobs
# Usage: bash slurm/submit_all.sh
set -e

mkdir -p slurm/slurm_logs results

SEEDS="42 123 456"
CATEGORIES="6 20 50 100"
DECAY_RATES="0.005 0.01 0.05"

COUNT=0

# ---------------------------------------------------------------
# Main sweep: each job runs all 10 algorithms
# ---------------------------------------------------------------
for K in $CATEGORIES; do
    for DECAY in $DECAY_RATES; do
        for SEED in $SEEDS; do
            sbatch -J sweep_k${K}_d${DECAY}_s${SEED} \
                slurm/run_experiment.sh $K $DECAY $SEED
            COUNT=$((COUNT + 1))
        done
    done
done

echo "Submitted $COUNT sweep jobs (each runs all 10 algorithms)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'grep -h \"\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/*.out | sort | tail -30'"
echo "  grep -l \"\\[FAILED\\]\" slurm/slurm_logs/*_*.out"
