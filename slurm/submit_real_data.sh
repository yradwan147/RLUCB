#!/bin/bash
# Submit real data experiments to IBEX cluster.
#
# Prerequisites:
#   1. Download data first: bash scripts/download_data.sh all
#   2. Data should be in data/duolingo/ and data/assistments/
#
# Total jobs: 2 datasets × 3 seeds = 6 jobs
set -e

mkdir -p slurm/slurm_logs results/real_data

SEEDS="42 123 456"
DATASETS="duolingo assistments"

COUNT=0

# ---------------------------------------------------------------
# Check data exists
# ---------------------------------------------------------------
for DS in $DATASETS; do
    if [ ! -d "data/$DS" ]; then
        echo "WARNING: data/$DS not found. Run: bash scripts/download_data.sh $DS"
    fi
done

# ---------------------------------------------------------------
# Submit real data evaluation jobs (both replay + fitted sim)
# ---------------------------------------------------------------
for DS in $DATASETS; do
    for SEED in $SEEDS; do
        sbatch -J rd_${DS}_s${SEED} \
            slurm/run_real_data.sh $DS $SEED both
        COUNT=$((COUNT + 1))
    done
done

echo "Submitted $COUNT real data jobs"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  grep -h '\\[SUCCESS\\]\\|\\[FAILED\\]' slurm/slurm_logs/rd_*.out"
