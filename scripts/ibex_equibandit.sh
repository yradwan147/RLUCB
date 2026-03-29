#!/bin/bash
# IBEX array job: EquiBandit knowledge + equity evaluation
# Covers K∈{6,20} × λ∈{0.005,0.01,0.05} × 10 seeds = 60 jobs
#
# Submit:  sbatch scripts/ibex_equibandit.sh
# Monitor: squeue -u $USER -n equibandit
# Collect: python scripts/collect_equibandit.py

#SBATCH --job-name=equibandit
#SBATCH --partition=batch
#SBATCH --array=1-60
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/slurm_logs/equibandit_%A_%a.out
#SBATCH --error=slurm/slurm_logs/equibandit_%A_%a.err

# ── Environment ──────────────────────────────────────────────────────────────
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate chessgcn

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/slurm_logs results/equibandit

# ── Config grid ──────────────────────────────────────────────────────────────
# 60 tasks: index = (config_idx - 1) * 10 + seed_idx  (1-indexed)
# config order: (K=6,l=0.005), (K=6,l=0.01), (K=6,l=0.05),
#               (K=20,l=0.005),(K=20,l=0.01),(K=20,l=0.05)
K_VALUES=(6 6 6 20 20 20)
L_VALUES=(0.005 0.01 0.05 0.005 0.01 0.05)
SEEDS=(42 123 456 789 1024 2048 3141 6283 9999 11111)

TASK_ID=$SLURM_ARRAY_TASK_ID          # 1..60
CONFIG_IDX=$(( (TASK_ID - 1) / 10 ))  # 0..5
SEED_IDX=$(( (TASK_ID - 1) % 10 ))    # 0..9

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
