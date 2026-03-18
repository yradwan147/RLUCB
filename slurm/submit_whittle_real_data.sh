#!/bin/bash
# Submit real data experiments with all 13 algorithms (including Whittle variants).
# Total: 6 jobs (2 datasets × 3 seeds)
set -e

mkdir -p slurm/slurm_logs results/real_data

SEEDS="42 123 456"
COUNT=0

for DS in duolingo assistments; do
    for SEED in $SEEDS; do
        sbatch -J wr_${DS}_s${SEED} \
            slurm/run_real_data.sh $DS $SEED both
        COUNT=$((COUNT + 1))
    done
done

echo "Submitted $COUNT real data jobs (all 13 algorithms)"
echo "Monitor: squeue -u \$USER"
