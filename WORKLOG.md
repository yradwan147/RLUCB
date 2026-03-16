# RLUCB NeurIPS Extension — Worklog

## Session 1 — 2026-03-16

### What was done

**Phase 0 — Project setup (COMPLETE)**
- GitHub remote verified (yradwan147/RLUCB)
- Updated `.gitignore` for results/, data/, slurm_logs/, *.pdf
- Created plans/, slurm/, results/, scripts/ directories

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

**Phase B — Simulation generalization (COMPLETE)**
- Added `MultiAlgorithmExperiment` class to `experiment/simulation.py`
- Added `MultiAlgorithmResults` with DataFrame export and CSV saving
- Supports N algorithm groups with identical initial conditions

**Phase C — Slurm integration (COMPLETE)**
- All jobs are CPU-only (no GPU needed)
- `slurm/run_experiment.sh` — runs all 10 algorithms per (K, λ, seed) combo
- `slurm/run_real_data.sh` — real data pipeline per (dataset, seed)
- `slurm/submit_all.sh` — 36 synthetic sweep jobs
- `slurm/submit_real_data.sh` — 6 real data jobs
- `slurm/submit_everything.sh` — 42 total jobs, installs deps first
- Uses shared `chessgcn` conda env + `pip install -r requirements.txt`

**Phase E — Real data pipeline (COMPLETE)**
- `experiment/real_data.py`: load Duolingo + ASSISTments, MLE fitting, replay eval
- `run_real_data.py`: CLI entry point
- `scripts/download_data.sh`: automated download (Harvard Dataverse + gdown)
- Duolingo file ID: 3091087, ASSISTments via gdown

**run_experiment.py updated**
- `--algorithm` flag for selecting specific algorithms
- `--all-algorithms` runs all 10 in one job
- `--csv` auto-exports results
- Backward compatible with legacy UCB vs Random mode

### Tests run locally
- All 10 selectors: smoke test (20 steps) ✓
- MultiAlgorithmExperiment: 10 algos × 10 students × 100 questions ✓
- CLI `--all-algorithms --csv`: 10 algos × 5 students × 50 questions ✓
- Scale test: 50 categories × 1000 questions ✓
- Real data pipeline end-to-end (synthetic traces): MLE fitting + replay ✓
- Params save/load round-trip ✓

### Jobs submitted on IBEX
- 42 jobs total (36 synthetic sweep + 6 real data)
- Synthetic grid: K ∈ {6, 20, 50, 100}, λ ∈ {0.005, 0.01, 0.05}, seeds ∈ {42, 123, 456}
- Real data: Duolingo + ASSISTments × 3 seeds
- Cancel command: `squeue -u $USER -o "%i %j" | grep -E "sweep_|rd_" | awk '{print $1}' | xargs -r scancel`

### What's next (after jobs finish)
- Collect and analyze results from IBEX
- Generate publication-quality visualizations
- Phase G: Theoretical analysis (F-UCB regret bounds, BKT-Bandit convergence)
- Phase H: Paper writing (last)
- May need additional ablation runs depending on results
