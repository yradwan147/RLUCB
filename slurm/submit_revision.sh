#!/bin/bash
# Submit ALL revision experiments to IBEX.
#
# B1: dTS baseline validation (6 jobs) — K={6,20} × 3 decay × seed=42
# B3: More seeds (6 jobs) — K={6,20} × 3 decay × 7 new seeds (batched)
#
# Total: ~12 jobs (well within 50-job limit)
# Note: B2 (λ sensitivity) requires code changes — run separately after
#
# Time limits: 8h for all (spacious to avoid failures)
set -e

mkdir -p slurm/slurm_logs results

COUNT=0

# ---------------------------------------------------------------
# B1: dTS baseline validation — run all 15 algorithms
# K=6 and K=20 at all 3 decay rates, seed=42
# ---------------------------------------------------------------
echo "--- B1: dTS baseline validation (6 jobs) ---"
for K in 6 20; do
    for DECAY in 0.005 0.01 0.05; do
        sbatch -J rev_k${K}_d${DECAY}_s42 \
            slurm/run_experiment.sh $K $DECAY 42
        COUNT=$((COUNT + 1))
    done
done

# ---------------------------------------------------------------
# B3: More seeds — K=6 and K=20, 7 additional seeds
# Each job runs all 15 algorithms
# ---------------------------------------------------------------
echo "--- B3: Additional seeds (6 jobs, 7 seeds each batched) ---"
NEW_SEEDS="789 1024 1337 2024 3141 4242 5555"

for K in 6 20; do
    for DECAY in 0.005 0.01 0.05; do
        # Submit one job per seed to stay within time limits
        for SEED in $NEW_SEEDS; do
            sbatch -J rev_k${K}_d${DECAY}_s${SEED} \
                slurm/run_experiment.sh $K $DECAY $SEED
            COUNT=$((COUNT + 1))
        done
    done
done

echo ""
echo "Submitted $COUNT revision jobs (15 algorithms each)"
echo "Monitor: squeue -u \$USER"
echo "Cancel: squeue -u \$USER -o '%i %j' | grep 'rev_' | awk '{print \$1}' | xargs -r scancel"
