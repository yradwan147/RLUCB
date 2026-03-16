#!/bin/bash
# Quick test: submit a small experiment to verify everything works on IBEX.
# Usage: bash slurm/submit_quick_test.sh
set -e

mkdir -p slurm/slurm_logs results

echo "Submitting quick test job (10 students, 6 categories, 200 questions)..."
sbatch -J test_quick \
    slurm/run_experiment.sh fucb 6 0.01 42 10 200

echo "Done. Check with: squeue -u \$USER"
echo "Output will be in: slurm/slurm_logs/test_quick_*.out"
