#!/bin/bash
# Re-run #2: failed K=50, K=100, and ASSISTments jobs.
#
# Fixes applied:
#   - OOM: disabled Student.history tracking (was storing 2×K floats per step)
#   - OOM: auto log_frequency caps at ~1000 data points
#   - ASSISTments: robust timestamp conversion (is_numeric_dtype check)
#
# Total: 18 synthetic + 3 ASSISTments = 21 jobs
set -e

mkdir -p slurm/slurm_logs results

SEEDS="42 123 456"
COUNT=0

# ---------------------------------------------------------------
# K=50 and K=100 synthetic sweep (K=6 and K=20 already done)
# ---------------------------------------------------------------
CATEGORIES="50 100"
DECAY_RATES="0.005 0.01 0.05"

echo "--- Re-running K=50,100 synthetic sweep (18 jobs) ---"
for K in $CATEGORIES; do
    for DECAY in $DECAY_RATES; do
        for SEED in $SEEDS; do
            sbatch -J sweep_k${K}_d${DECAY}_s${SEED} \
                slurm/run_experiment.sh $K $DECAY $SEED
            COUNT=$((COUNT + 1))
        done
    done
done

# ---------------------------------------------------------------
# ASSISTments real data
# ---------------------------------------------------------------
echo "--- Re-running ASSISTments real data (3 jobs) ---"
for SEED in $SEEDS; do
    sbatch -J rd_assistments_s${SEED} \
        slurm/run_real_data.sh assistments $SEED both
    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted $COUNT re-run jobs"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'grep -h \"\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/*.out | sort | tail -30'"
