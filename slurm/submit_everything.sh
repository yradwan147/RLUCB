#!/bin/bash
# Submit ALL experiments to IBEX cluster — synthetic sweep + real data.
#
# Prerequisites:
#   1. Install deps: conda activate chessgcn && pip install -r requirements.txt
#   2. Real data: bash scripts/download_data.sh all
#
# This submits:
#   - 36 synthetic sweep jobs (submit_all.sh) — each runs all 10 algorithms
#   - 6 real data jobs (submit_real_data.sh)
#   Total: 42 jobs
set -e

echo "============================================="
echo "Submitting ALL experiments to IBEX"
echo "============================================="
echo ""

# Install dependencies into chessgcn env
echo "--- Installing dependencies into chessgcn ---"
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate chessgcn
pip install -r requirements.txt --quiet
echo "Dependencies installed."
echo ""

# Synthetic sweep
echo "--- Synthetic sweep (36 jobs, each runs all 10 algos) ---"
bash slurm/submit_all.sh
echo ""

# Real data
echo "--- Real data experiments (6 jobs) ---"
bash slurm/submit_real_data.sh
echo ""

echo "============================================="
echo "All jobs submitted!"
echo "============================================="
echo ""
echo "Monitor all jobs:"
echo "  squeue -u \$USER | wc -l"
echo "  watch -n 60 'grep -ch \"\\[SUCCESS\\]\" slurm/slurm_logs/*.out | paste -sd+ | bc'"
