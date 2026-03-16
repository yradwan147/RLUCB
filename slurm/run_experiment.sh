#!/bin/bash --login
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/slurm_logs/%x_%J.out

# Runs ALL 10 algorithms for a single (K, decay, seed) config.
# Usage: sbatch -J <job_name> slurm/run_experiment.sh <num_categories> <decay_rate> <seed> [num_students] [questions]

NUM_CATEGORIES=${1:?"Num categories required (e.g., 6, 20, 50, 100)"}
DECAY_RATE=${2:?"Decay rate required (e.g., 0.001, 0.005, 0.01, 0.05, 0.1)"}
SEED=${3:?"Seed required (e.g., 42, 123, 456, 789, 1024)"}
NUM_STUDENTS=${4:-100}
QUESTIONS=${5:-10000}

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate chessgcn

cd $SLURM_SUBMIT_DIR

echo "[STARTED] K=$NUM_CATEGORIES decay=$DECAY_RATE seed=$SEED students=$NUM_STUDENTS questions=$QUESTIONS"
echo "[STARTED] $(date)"

python run_experiment.py \
    --all-algorithms \
    --num-students $NUM_STUDENTS \
    --categories $NUM_CATEGORIES \
    --questions $QUESTIONS \
    --decay-rate $DECAY_RATE \
    --seed $SEED \
    --output-dir results/k${NUM_CATEGORIES}_d${DECAY_RATE}_s${SEED} \
    --no-show \
    --csv

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] K=$NUM_CATEGORIES decay=$DECAY_RATE seed=$SEED"
else
    echo "[FAILED] K=$NUM_CATEGORIES decay=$DECAY_RATE seed=$SEED exit_code=$EXIT_CODE"
fi
echo "[FINISHED] $(date)"
