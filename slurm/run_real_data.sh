#!/bin/bash --login
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/slurm_logs/%x_%J.out

# Usage: sbatch -J <job_name> slurm/run_real_data.sh <dataset> <seed> [replay_only|sim_only|both]

DATASET=${1:?"Dataset required (duolingo or assistments)"}
SEED=${2:?"Seed required"}
MODE=${3:-both}  # replay_only, sim_only, or both

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate chessgcn

cd $SLURM_SUBMIT_DIR

echo "[STARTED] dataset=$DATASET seed=$SEED mode=$MODE"
echo "[STARTED] $(date)"

EXTRA_ARGS=""
if [ "$MODE" = "replay_only" ]; then
    EXTRA_ARGS="--replay-only"
elif [ "$MODE" = "sim_only" ]; then
    EXTRA_ARGS="--sim-only"
fi

python run_real_data.py \
    --dataset $DATASET \
    --data-dir data/ \
    --output-dir results/real_data/${DATASET}_s${SEED} \
    --max-students-fit 500 \
    --max-students-replay 500 \
    --max-interactions 1000 \
    --seed $SEED \
    --verbose \
    $EXTRA_ARGS

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] dataset=$DATASET seed=$SEED mode=$MODE"
else
    echo "[FAILED] dataset=$DATASET seed=$SEED mode=$MODE exit_code=$EXIT_CODE"
fi
echo "[FINISHED] $(date)"
