#!/bin/bash
# Submit ALL experiments to IBEX cluster — synthetic sweep + real data.
#
# Prerequisites:
#   1. conda env: conda env create -f environment.yml
#   2. Real data: bash scripts/download_data.sh all
#
# This submits:
#   - 1000 synthetic sweep jobs (submit_all.sh)
#   - 10 real data jobs (submit_real_data.sh)
#   Total: 1010 jobs
set -e

echo "============================================="
echo "Submitting ALL experiments to IBEX"
echo "============================================="
echo ""

# Synthetic sweep
echo "--- Synthetic sweep (1000 jobs) ---"
bash slurm/submit_all.sh
echo ""

# Real data
echo "--- Real data experiments (10 jobs) ---"
bash slurm/submit_real_data.sh
echo ""

echo "============================================="
echo "All jobs submitted!"
echo "============================================="
echo ""
echo "Monitor all jobs:"
echo "  squeue -u \$USER | wc -l"
echo "  watch -n 60 'grep -ch \"\\[SUCCESS\\]\" slurm/slurm_logs/*.out | paste -sd+ | bc'"
