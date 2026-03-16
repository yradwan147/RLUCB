#!/bin/bash
# Submit all main sweep experiments to IBEX cluster.
# Usage: bash slurm/submit_all.sh
#
# Total jobs: 10 algorithms × 4 category counts × 5 decay rates × 5 seeds = 1000
set -e

mkdir -p slurm/slurm_logs results

SEEDS="42 123 456 789 1024"
ALGORITHMS="random ucb1 fucb bkt_bandit bkt_thompson thompson epsilon_greedy sw_ucb leitner oracle"
CATEGORIES="6 20 50 100"
DECAY_RATES="0.001 0.005 0.01 0.05 0.1"

COUNT=0

# ---------------------------------------------------------------
# Main sweep: all algorithms × categories × decay rates × seeds
# ---------------------------------------------------------------
for ALGO in $ALGORITHMS; do
    for K in $CATEGORIES; do
        for DECAY in $DECAY_RATES; do
            for SEED in $SEEDS; do
                # Short prefix for job naming
                case $ALGO in
                    random)         PFX="rnd" ;;
                    ucb1)           PFX="ucb" ;;
                    fucb)           PFX="fucb" ;;
                    bkt_bandit)     PFX="bktb" ;;
                    bkt_thompson)   PFX="bktt" ;;
                    thompson)       PFX="ts" ;;
                    epsilon_greedy) PFX="eg" ;;
                    sw_ucb)         PFX="swu" ;;
                    leitner)        PFX="ltn" ;;
                    oracle)         PFX="orc" ;;
                esac
                sbatch -J ${PFX}_k${K}_d${DECAY}_s${SEED} \
                    slurm/run_experiment.sh $ALGO $K $DECAY $SEED
                COUNT=$((COUNT + 1))
            done
        done
    done
done

echo "Submitted $COUNT main sweep jobs"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'grep -h \"\\[SUCCESS\\]\\|\\[FAILED\\]\" slurm/slurm_logs/*.out | sort | tail -30'"
echo "  grep -l \"\\[FAILED\\]\" slurm/slurm_logs/*_*.out"
