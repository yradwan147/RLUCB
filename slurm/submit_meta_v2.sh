#!/bin/bash
# Submit MetaSelector v2 experiments — blended ΔK + correctness reward.
# All 14 algorithms on synthetic + real data.
# Total: 42 jobs
set -e

mkdir -p slurm/slurm_logs results

SEEDS="42 123 456"
COUNT=0

# Synthetic sweep (36 jobs)
echo "--- Synthetic sweep (36 jobs, 14 algorithms each) ---"
for K in 6 20 50 100; do
    for DECAY in 0.005 0.01 0.05; do
        for SEED in $SEEDS; do
            sbatch -J mv2_k${K}_d${DECAY}_s${SEED} \
                slurm/run_experiment.sh $K $DECAY $SEED
            COUNT=$((COUNT + 1))
        done
    done
done

# Real data (6 jobs) — now uses all algorithms from registry
echo "--- Real data (6 jobs, all algorithms) ---"
for DS in duolingo assistments; do
    for SEED in $SEEDS; do
        sbatch -J mv2_rd_${DS}_s${SEED} \
            slurm/run_real_data.sh $DS $SEED both
        COUNT=$((COUNT + 1))
    done
done

echo ""
echo "Submitted $COUNT jobs (14 algorithms including MetaSelector v2)"
echo "Monitor: squeue -u \$USER"
echo "Cancel: squeue -u \$USER -o '%i %j' | grep 'mv2_' | awk '{print \$1}' | xargs -r scancel"
