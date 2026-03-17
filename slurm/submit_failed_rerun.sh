#!/bin/bash
# Re-run failed jobs from run 1.
#
# Fixes applied:
#   - OOM: auto log_frequency (log every N steps, ~1000 data points max)
#   - OOM: increased memory to 32G
#   - ASSISTments: fixed datetime string timestamp parsing
#
# Failed jobs:
#   - All K≥20 sweep jobs (OOM killed, exit code 137)
#   - All ASSISTments real data jobs (timestamp parse error)
#
# Total: 27 synthetic + 3 ASSISTments = 30 jobs
set -e

mkdir -p slurm/slurm_logs results

SEEDS="42 123 456"
COUNT=0

# ---------------------------------------------------------------
# Synthetic sweep: K≥20 (K=6 already succeeded)
# ---------------------------------------------------------------
CATEGORIES="20 50 100"
DECAY_RATES="0.005 0.01 0.05"

echo "--- Re-running K≥20 synthetic sweep (27 jobs) ---"
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
# ASSISTments real data (Duolingo already succeeded)
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
