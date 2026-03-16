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

---

## Paper References (for Phase H)

### Already in `paper/refs.bib`
- Auer et al. 2002 — UCB1 finite-time analysis (foundational)
- Lattimore & Szepesvári 2020 — Bandit Algorithms textbook
- Ebbinghaus 1885 — Forgetting curve (foundational)
- Murre & Dros 2015 — Ebbinghaus replication
- Corbett & Anderson 1994 — Bayesian Knowledge Tracing (BKT)
- Piech et al. 2015 — Deep Knowledge Tracing (NeurIPS)
- Yudelson et al. 2013 — Individualized BKT
- Clément et al. 2015 — Multi-armed bandits for ITS
- Liu & Koedinger 2014 — Trading off learning with bandits
- Rafferty et al. 2019 — Statistical consequences of MAB in education
- Doroudi et al. 2019 — Where's the reward? (RL for education)
- Settles & Meeder 2016 — Duolingo half-life regression (dataset source)
- Leitner 1972 — Leitner box system
- Pavlik & Anderson 2005 — Practice/forgetting activation model
- VanLehn 2011 — ITS effectiveness review
- Towers et al. 2024 — Gymnasium

### New references to add for extended paper

**Non-stationary bandits (theory for F-UCB):**
- Garivier & Moulines 2011 — "On Upper-Confidence Bound Policies for Switching Bandit Problems" (SW-UCB, foundation for our non-stationarity argument)
- Besbes et al. 2014 — "Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards" (variation budget regret bounds)
- Russac et al. 2019 — "Weighted Linear Bandits for Non-Stationary Environments" (D-LinUCB)

**Thompson Sampling:**
- Thompson 1933 — "On the likelihood that one unknown probability exceeds another" (original)
- Chapelle & Li 2011 — "An Empirical Evaluation of Thompson Sampling" (empirical comparison vs UCB)
- Agrawal & Goyal 2012 — "Analysis of Thompson Sampling for the Multi-Armed Bandit Problem" (regret bounds for TS)
- Russo et al. 2018 — "A Tutorial on Thompson Sampling" (comprehensive survey)

**Variance-dependent / contextual bandits (NeurIPS 2024):**
- Zhou & Tan 2024 — "How Does Variance Shape the Regret in Contextual Bandits?" (NeurIPS 2024)
- He et al. 2024 — "Cert-LSVI-UCB: Constant instance-dependent regret bounds in RL" (NeurIPS 2024)

**Knowledge tracing + RL (competing methods):**
- Zhang et al. 2025 — "Integrating RL with Dynamic Knowledge Tracing (RL-DKT)" (Nature Scientific Reports)
- Shi et al. 2023 — "Adaptive Learning Path Navigation (ALPN): AKT + Entropy-enhanced PPO" (arXiv)
- Ghosh et al. 2020 — "Contextual bandits with latent confounders" (connections to hidden knowledge state)

**Spaced repetition theory:**
- Pimsleur 1967 — "A Memory Schedule" (graduated interval recall)
- Wozniak & Gorzelanczyk 1995 — "Two components of long-term memory" (already in bib)
- Reddy et al. 2016 — "Unbounded Human Learning: Optimal Scheduling for Spaced Repetition" (KDD)

**Educational data mining:**
- Settles et al. 2018 — "Second Language Acquisition Modeling (SLAM)" shared task (Duolingo dataset)
- Feng et al. 2009 — "ASSISTments dataset" (dataset source)
- Baker & Inventado 2014 — "Educational Data Mining and Learning Analytics" (survey)

**Bayesian experimental design (for BKT-Bandit theory):**
- Chaloner & Verdinelli 1995 — "Bayesian Experimental Design: A Review"
- Lindley 1956 — "On a Measure of Information Provided by an Experiment" (information gain)

**Regret lower bounds:**
- Lai & Robbins 1985 — "Asymptotically efficient adaptive allocation rules" (foundational lower bound)

**Real data baselines:**
- Choffin et al. 2019 — "DAS3H: Modeling Student Learning and Forgetting" (KDD, combines skills + forgetting)
- Wilson et al. 2016 — "Back to the basics: Bayesian extensions of IRT" (Bayesian student modeling)
