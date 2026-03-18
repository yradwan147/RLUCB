#!/bin/bash
# Rerun real data experiments with fixed pipeline.
#
# Fixes:
#   - Replay now uses real deltas (not step-counter) for forgetting
#   - Time unit stored in fitted params, consistent between fit and replay
#   - Multiple optimizer restarts (5) to avoid local optima
#   - Removed gradient-breaking clipping inside objective
#   - Deterministic selector seeding (no hash())
#
# Also reruns missing K=50 d=0.05 s=456.
#
# Total: 6 real data + 1 synthetic = 7 jobs
set -e

mkdir -p slurm/slurm_logs results

SEEDS="42 123 456"
COUNT=0

# ---------------------------------------------------------------
# Real data (Duolingo + ASSISTments) with fixed pipeline
# ---------------------------------------------------------------
echo "--- Submitting fixed real data jobs (6 jobs) ---"
for DS in duolingo assistments; do
    for SEED in $SEEDS; do
        sbatch -J rd_fixed_${DS}_s${SEED} \
            slurm/run_real_data.sh $DS $SEED both
        COUNT=$((COUNT + 1))
    done
done

# ---------------------------------------------------------------
# Missing synthetic: K=50 d=0.05 s=456
# ---------------------------------------------------------------
echo "--- Submitting missing K=50 d=0.05 s=456 ---"
sbatch -J sweep_k50_d0.05_s456 \
    slurm/run_experiment.sh 50 0.05 456
COUNT=$((COUNT + 1))

echo ""
echo "Submitted $COUNT jobs"
echo "Monitor: squeue -u \$USER"
