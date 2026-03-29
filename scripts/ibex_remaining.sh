#!/bin/bash
# Submit ALL remaining experiments to IBEX.
#
# 1. EquiBandit at K=50,100 (18 jobs)           — fills scaling + consistency tables
# 2. Real data: missing algorithms (6 jobs)      — adds EquiBandit, dTS, Lookahead to real data
# 3. Sensitivity: extra seeds (10 jobs)           — strengthens sensitivity analysis
#
# Total: 34 jobs
#
# Usage:  bash scripts/ibex_remaining.sh
# Monitor: squeue -u $USER
set -e

mkdir -p slurm/slurm_logs results/equibandit results/sensitivity

COUNT=0

# ═══════════════════════════════════════════════════════════════════
# 1. EquiBandit at K=50,100 (18 jobs via array)
# ═══════════════════════════════════════════════════════════════════
echo "--- 1. EquiBandit K=50,100 (18 array jobs) ---"
sbatch scripts/ibex_equibandit_large.sh
COUNT=$((COUNT + 18))

# ═══════════════════════════════════════════════════════════════════
# 2. Real data: EquiBandit + Discounted TS + Lookahead Oracle
#    Run fitted-sim only (not replay) for missing algorithms
#    2 datasets × 3 seeds = 6 jobs
# ═══════════════════════════════════════════════════════════════════
echo "--- 2. Real data: missing algorithms (6 jobs) ---"
for DS in duolingo assistments; do
    for SEED in 42 123 456; do
        sbatch -J rd2_${DS}_s${SEED} \
            --partition=batch \
            --cpus-per-task=4 \
            --mem=32G \
            --time=8:00:00 \
            --output=slurm/slurm_logs/rd2_${DS}_s${SEED}_%J.out \
            --error=slurm/slurm_logs/rd2_${DS}_s${SEED}_%J.err \
            --wrap="
eval \"\$(~/miniconda3/bin/conda shell.bash hook)\"
conda activate chessgcn
cd \$SLURM_SUBMIT_DIR
echo '[STARTED] dataset=$DS seed=$SEED algos=equi_bandit,discounted_ts,lookahead_oracle'
python run_real_data.py \
    --dataset $DS \
    --data-dir data/ \
    --algorithms equi_bandit discounted_ts lookahead_oracle \
    --output-dir results/real_data/${DS}_s${SEED}_extra \
    --max-students-fit 500 \
    --max-students-replay 500 \
    --max-interactions 1000 \
    --seed $SEED \
    --sim-only \
    --verbose
echo '[FINISHED] dataset=$DS seed=$SEED'
"
        COUNT=$((COUNT + 1))
    done
done

# ═══════════════════════════════════════════════════════════════════
# 3. Sensitivity: 2 extra seeds (seeds 123, 456)
#    5 λ-ratios × 2 seeds = 10 jobs
# ═══════════════════════════════════════════════════════════════════
echo "--- 3. Sensitivity extra seeds (10 jobs) ---"
for ALGO_DECAY in 0.005 0.008 0.01 0.012 0.02; do
    for SEED in 123 456; do
        sbatch -J sens2_ad${ALGO_DECAY}_s${SEED} \
            slurm/run_sensitivity.sh 0.01 $ALGO_DECAY $SEED
        COUNT=$((COUNT + 1))
    done
done

echo ""
echo "============================================="
echo "Submitted $COUNT jobs total"
echo "============================================="
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Check:    grep -h '\\[SUCCESS\\]\\|\\[FAILED\\]' slurm/slurm_logs/equibandit_lg_*.out slurm/slurm_logs/rd2_*.out slurm/slurm_logs/sens2_*.out"
