# RLUCB NeurIPS Extension — Worklog

## Session 1 — 2026-03-16

### What was done
- Explored full repo and researched NeurIPS landscape for adaptive learning bandits
- Designed extension plan: F-UCB + BKT-Bandit algorithms, 10 baselines, scaling experiments, real data, theory
- Set up project structure: plans/, slurm/, results/, environment.yml
- Updated .gitignore for experiment outputs and large files
- Saved plan to plans/neurips_extension_plan.md
- Set up memory files for cross-session context

**Phase A — New selectors (COMPLETE)**
- Implemented 8 new selectors in `experiment/selectors.py`:
  - `FUCBSelector` — Forgetting-aware UCB with time-decay + urgency
  - `BKTBanditSelector` — Bayesian knowledge-state bandit with Beta posteriors
  - `BKTThompsonSelector` — Thompson Sampling with forgetting-aware posteriors
  - `ThompsonSelector` — Standard Thompson Sampling (no forgetting)
  - `EpsilonGreedySelector` — ε-greedy with ε=0.1
  - `SlidingWindowUCBSelector` — SW-UCB with configurable window
  - `LeitnerSelector` — Spaced repetition (Leitner box system)
  - `OracleSelector` — Cheats with true knowledge (upper bound)
- Added `SELECTOR_REGISTRY` and `create_selector()` factory function
- All 10 selectors pass smoke tests (20 steps, valid outputs)

**Phase B — Simulation generalization (COMPLETE)**
- Added `MultiAlgorithmExperiment` class to `experiment/simulation.py`
- Added `MultiAlgorithmResults` with DataFrame export and CSV saving
- Supports N algorithm groups with identical initial conditions
- Oracle selector correctly receives knowledge reference

**Phase C — Slurm integration (COMPLETE)**
- Created `slurm/run_experiment.sh` — SLURM runner with positional args
- Created `slurm/submit_all.sh` — orchestrator for 1000-job sweep
- Created `slurm/submit_ablations.sh` — ablation submission
- Created `slurm/submit_quick_test.sh` — single test job
- Created `environment.yml` for cluster conda env

**run_experiment.py updated**
- Added `--algorithm` flag for multi-algorithm mode
- Added `--all-algorithms` flag
- Added `--num-students` and `--output-dir` aliases for CLI compatibility
- Backward compatible: no --algorithm falls back to legacy UCB vs Random

**Phase E — Real data pipeline (COMPLETE)**
- Created `experiment/real_data.py` (~500 lines):
  - `load_duolingo()` — loads Duolingo Spaced Repetition 2016 (13M traces)
  - `load_assistments()` — loads ASSISTments 2012-2013 (2.5M interactions)
  - `fit_student_model()` — MLE fitting of α, β, λ, base, k0 from real data
  - `replay_evaluate()` — model-based replay evaluation under different bandit policies
  - `run_fitted_simulation()` — synthetic simulation with fitted parameters
  - `run_real_data_experiment()` — full pipeline: load → fit → replay → simulate
  - `FittedParams` dataclass with save/load (JSON)
  - `ReplayResult` with trajectory data for plotting
- Created `run_real_data.py` — CLI entry point with full argument parsing
- Created `scripts/download_data.sh` — automated data download from Harvard Dataverse
- Created SLURM scripts:
  - `slurm/run_real_data.sh` — SLURM runner (64G RAM, 24h wall time)
  - `slurm/submit_real_data.sh` — submits 10 jobs (2 datasets × 5 seeds)
  - `slurm/submit_everything.sh` — submits ALL experiments (1010 jobs total)

### Tests run
- Smoke test: all 10 selectors produce valid outputs (20 steps)
- MultiAlgorithmExperiment: 10 algorithms × 10 students × 100 questions — passes
- CLI test: `--algorithm random ucb1 fucb bkt_bandit oracle` — 200 questions — passes
- Scale test: 50 categories × 1000 questions — passes
- CSV export: correct shape (5006 rows for 5 algos × 1001 timesteps)
- Real data pipeline end-to-end test with synthetic traces:
  - MLE fitting: recovered α=0.135 (true 0.1), β=0.037 (true 0.02) — reasonable
  - Replay evaluation: 5 algorithms on 10 students × 80 interactions — passes
  - UCB1 weakest=0.270 > random weakest=0.166 — expected pattern holds
  - Params save/load round-trip: OK

### What's next
- Push to GitHub, pull on IBEX
- Download real data on IBEX: `bash scripts/download_data.sh all`
- Submit everything: `bash slurm/submit_everything.sh` (1010 jobs)
- Phase G: Theoretical analysis (regret bounds for F-UCB, BKT-Bandit convergence)
