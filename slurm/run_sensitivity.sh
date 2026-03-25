#!/bin/bash --login
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/slurm_logs/%x_%J.out

# Runs λ mis-specification sensitivity test.
# Usage: sbatch -J <name> slurm/run_sensitivity.sh <true_decay> <algo_decay> <seed>

TRUE_DECAY=${1:?"True decay rate required"}
ALGO_DECAY=${2:?"Algorithm decay rate required"}
SEED=${3:?"Seed required"}

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate chessgcn

cd $SLURM_SUBMIT_DIR

echo "[STARTED] true_decay=$TRUE_DECAY algo_decay=$ALGO_DECAY seed=$SEED"
echo "[STARTED] $(date)"

python run_experiment.py \
    --all-algorithms \
    --num-students 100 \
    --categories 6 \
    --questions 10000 \
    --decay-rate $TRUE_DECAY \
    --algorithm-decay-rate $ALGO_DECAY \
    --seed $SEED \
    --output-dir results/sensitivity/true${TRUE_DECAY}_algo${ALGO_DECAY}_s${SEED} \
    --no-show \
    --csv

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] true=$TRUE_DECAY algo=$ALGO_DECAY seed=$SEED"
else
    echo "[FAILED] true=$TRUE_DECAY algo=$ALGO_DECAY seed=$SEED exit_code=$EXIT_CODE"
fi
echo "[FINISHED] $(date)"
