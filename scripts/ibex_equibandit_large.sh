#!/bin/bash
# IBEX array job: EquiBandit at large K (50, 100)
# Covers K∈{50,100} × λ∈{0.005,0.01,0.05} × 3 seeds = 18 jobs
#
# Submit:  sbatch scripts/ibex_equibandit_large.sh
# Monitor: squeue -u $USER -n equibandit_lg
# Collect: python scripts/collect_equibandit.py

#SBATCH --job-name=equibandit_lg
#SBATCH --partition=batch
#SBATCH --array=1-18
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/slurm_logs/equibandit_lg_%A_%a.out
#SBATCH --error=slurm/slurm_logs/equibandit_lg_%A_%a.err

# ── Environment ──────────────────────────────────────────────────────────────
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate chessgcn

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/slurm_logs results/equibandit

# ── Config grid ──────────────────────────────────────────────────────────────
# 18 tasks: 6 configs × 3 seeds
# config order: (K=50,l=0.005), (K=50,l=0.01), (K=50,l=0.05),
#               (K=100,l=0.005),(K=100,l=0.01),(K=100,l=0.05)
K_VALUES=(50 50 50 100 100 100)
L_VALUES=(0.005 0.01 0.05 0.005 0.01 0.05)
SEEDS=(42 123 456)

TASK_ID=$SLURM_ARRAY_TASK_ID         # 1..18
CONFIG_IDX=$(( (TASK_ID - 1) / 3 ))  # 0..5
SEED_IDX=$(( (TASK_ID - 1) % 3 ))    # 0..2

K=${K_VALUES[$CONFIG_IDX]}
LAMBDA=${L_VALUES[$CONFIG_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUTDIR="results/equibandit/K${K}_L${LAMBDA}"
mkdir -p "$OUTDIR"

echo "Task ${TASK_ID}: K=${K}  lambda=${LAMBDA}  seed=${SEED}"

# ── Run EquiBandit vs BKT-Bandit vs Random ────────────────────────────────
python run_experiment.py \
    --categories    "${K}" \
    --decay-rate    "${LAMBDA}" \
    --seed          "${SEED}" \
    --students      100 \
    --questions     10000 \
    --algorithm     equi_bandit bkt_bandit random \
    --output-dir    "${OUTDIR}/seed${SEED}" \
    --no-show \
    --quiet \
    --csv

echo "Done: K=${K} lambda=${LAMBDA} seed=${SEED}"
