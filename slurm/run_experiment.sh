#!/bin/bash --login
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/slurm_logs/%x_%J.out

# Usage: sbatch -J <job_name> slurm/run_experiment.sh <algorithm> <num_categories> <decay_rate> <seed> [num_students] [questions]

ALGORITHM=${1:?"Algorithm required (e.g., random, ucb1, fucb, bkt_bandit, ...)"}
NUM_CATEGORIES=${2:?"Num categories required (e.g., 6, 20, 50, 100)"}
DECAY_RATE=${3:?"Decay rate required (e.g., 0.001, 0.005, 0.01, 0.05, 0.1)"}
SEED=${4:?"Seed required (e.g., 42, 123, 456, 789, 1024)"}
NUM_STUDENTS=${5:-100}
QUESTIONS=${6:-10000}

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate rlucb

cd $SLURM_SUBMIT_DIR

echo "[STARTED] algo=$ALGORITHM K=$NUM_CATEGORIES decay=$DECAY_RATE seed=$SEED students=$NUM_STUDENTS questions=$QUESTIONS"
echo "[STARTED] $(date)"

python run_experiment.py \
    --algorithm $ALGORITHM \
    --num-students $NUM_STUDENTS \
    --categories $NUM_CATEGORIES \
    --questions $QUESTIONS \
    --decay-rate $DECAY_RATE \
    --seed $SEED \
    --output-dir results/${ALGORITHM}_k${NUM_CATEGORIES}_d${DECAY_RATE}_s${SEED} \
    --no-show \
    --quiet

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] algo=$ALGORITHM K=$NUM_CATEGORIES decay=$DECAY_RATE seed=$SEED"
else
    echo "[FAILED] algo=$ALGORITHM K=$NUM_CATEGORIES decay=$DECAY_RATE seed=$SEED exit_code=$EXIT_CODE"
fi
echo "[FINISHED] $(date)"
