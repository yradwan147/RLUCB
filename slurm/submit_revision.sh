#!/bin/bash
# Submit ALL revision experiments to IBEX.
#
# B1: dTS baseline validation (6 jobs)
# B2: λ mis-specification sensitivity (5 jobs)
# B3: More seeds for K=6,20 (42 jobs)
#
# Total: 53 jobs
# Time limits: 8h for experiments, 4h for sensitivity
set -e

mkdir -p slurm/slurm_logs results results/sensitivity

COUNT=0

# ---------------------------------------------------------------
# B1: dTS baseline validation — K=6,20 × 3 decay × seed=42
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
# B2: λ sensitivity — true λ=0.01, algorithm λ varies
# K=6, seed=42
# ---------------------------------------------------------------
echo "--- B2: λ sensitivity (5 jobs) ---"
for ALGO_DECAY in 0.005 0.008 0.01 0.012 0.02; do
    sbatch -J sens_ad${ALGO_DECAY}_s42 \
        slurm/run_sensitivity.sh 0.01 $ALGO_DECAY 42
    COUNT=$((COUNT + 1))
done

# ---------------------------------------------------------------
# B3: More seeds — K=6,20 × 3 decay × 7 new seeds
# ---------------------------------------------------------------
echo "--- B3: Additional seeds (42 jobs) ---"
for K in 6 20; do
    for DECAY in 0.005 0.01 0.05; do
        for SEED in 789 1024 1337 2024 3141 4242 5555; do
            sbatch -J rev_k${K}_d${DECAY}_s${SEED} \
                slurm/run_experiment.sh $K $DECAY $SEED
            COUNT=$((COUNT + 1))
        done
    done
done

# ---------------------------------------------------------------
# C1: LookaheadOracle validation — K=6 × 3 decay × seed=42
# ---------------------------------------------------------------
echo "--- C1: LookaheadOracle (3 jobs) ---"
for DECAY in 0.005 0.01 0.05; do
    sbatch -J rev_la_k6_d${DECAY}_s42 \
        slurm/run_experiment.sh 6 $DECAY 42
    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted $COUNT revision jobs"
echo "Monitor: squeue -u \$USER"
echo "Cancel: squeue -u \$USER -o '%i %j' | grep -E 'rev_|sens_' | awk '{print \$1}' | xargs -r scancel"
