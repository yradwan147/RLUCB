# Plan: Extend RLUCB to NeurIPS-Level Paper

## Context

The current RLUCB project applies UCB1 to adaptive question selection for simulated students, showing that UCB achieves 60.7% better performance in weak categories early on but converges to random selection long-term. While the result is interesting, the paper currently falls short of NeurIPS standards because: (1) synthetic-only evaluation, (2) UCB1 is not novel, (3) only 2 baselines, (4) no theoretical contribution, (5) toy scale (6 categories), (6) no knowledge tracing. This plan addresses all six gaps.

## Core Thesis

**"Standard bandits fail in educational settings because they ignore forgetting. We formalize why, propose two complementary solutions (frequentist and Bayesian), prove guarantees, and validate on real data."**

The existing crossover phenomenon (UCB wins early, loses late) is the motivating empirical observation — it reveals that UCB1's stationarity assumption is violated by forgetting dynamics. This reframing turns a limitation into the paper's central contribution.

---

## Part 1: New Algorithms (extend `experiment/selectors.py`)

### 1a. F-UCB (Forgetting-aware UCB) — Primary contribution
- Replace static `(1 - correctness_rate)` with a **time-decayed** estimate:
  ```
  score(i) = (1 - r_i) * exp(-λ * t_i) + γ * (1 - exp(-λ * t_i)) + c * sqrt(ln(N) / n_i)
  ```
  where `t_i` = `time_since_exposure[i]` (already tracked in `Student`)
- The forgetting-urgency term `γ * (1 - exp(-λ * t_i))` increases score for categories not recently visited
- Implement as `FUCBSelector(BaseSelector)` in `selectors.py`

### 1b. BKT-Bandit (Bayesian Knowledge-State Bandit) — Secondary contribution
- Maintain Beta(α_i, β_i) posterior per category for latent knowledge
- Selection: `argmax_i [(1 - mean(Beta_i)) + c * std(Beta_i)]`
- Between observations, apply forgetting transition that **flattens the posterior** (increases variance)
- Implement as `BKTBanditSelector(BaseSelector)` in `selectors.py`

### 1c. BKT-Thompson Sampling variant
- Sample k_i ~ Beta_i for each category, select `argmax_i (1 - k_i)`
- Implement as `BKTThompsonSelector(BaseSelector)` in `selectors.py`

### 1d. Additional baselines (all as `BaseSelector` subclasses)
- `ThompsonSelector` — standard Thompson Sampling (Beta-Bernoulli)
- `EpsilonGreedySelector` — ε-greedy with ε=0.1
- `SlidingWindowUCBSelector` — SW-UCB (Garivier & Moulines 2011), window W
- `LeitnerSelector` — spaced repetition heuristic (Leitner box system)
- `OracleSelector` — cheats by reading true knowledge vector (upper bound)

**Total: 8 new selectors + 2 existing = 10 algorithms compared**

---

## Part 2: Theoretical Contributions

### 2a. Regret bound for F-UCB
- Define regret relative to oracle that always quizzes the category with highest marginal knowledge gain
- Prove: `R(T) = O(sqrt(K * T * (1 + λ)))` where K = categories, λ = forgetting rate
- Key insight: forgetting rate λ increases regret because it creates non-stationarity
- Use techniques from SW-UCB analysis + structure of asymptotic learning model

### 2b. Convergence guarantee for BKT-Bandit
- Show posterior concentrates at rate depending on forgetting dynamics
- Connect to information-theoretic experimental design (mutual information maximization)
- Prove: BKT-Bandit identifies the weakest category in O(K log K / Δ²) steps where Δ = knowledge gap

### 2c. Lower bound
- Prove Ω(sqrt(K * T)) regret in the forgetting setting
- Shows F-UCB is near-optimal up to the (1+λ) factor

---

## Part 3: Extended Experiments

### 3a. Scaling experiments (modify `experiment/config.py`)
- Add support for num_categories in {6, 20, 50, 100}
- Add parameter sweep infrastructure for decay_rate in {0.001, 0.005, 0.01, 0.05, 0.1}
- Run all 10 algorithms × 5 forgetting rates × 4 category counts = 200 configurations

### 3b. Ablation studies
- F-UCB without forgetting-urgency term (just time-decay)
- F-UCB without time-decay (just forgetting-urgency)
- BKT-Bandit with fixed vs. adaptive posterior flattening
- Sensitivity to exploration parameter c

### 3c. Heterogeneous students
- Extend `Student.__init__` to support per-student learning rates and forgetting rates
- Sample from a population distribution: α_i ~ N(0.12, 0.03), λ_i ~ N(0.01, 0.003)
- Test which algorithms adapt best to student heterogeneity

### 3d. Cold-start analysis
- Measure "time to identify weakest category" for each algorithm
- BKT-Bandit should win because posterior uncertainty drives initial exploration

### 3e. Real data validation
- **Duolingo SLAM dataset** (already cited in bibliography): 13M exercises, 6K+ users
  - Fit student model parameters via MLE from historical trajectories
  - Run replay-based counterfactual evaluation
- **ASSISTments 2012**: Math tutoring, ~300 skills, timestamped
  - Same approach: fit model, replay evaluation
- Create `experiment/real_data.py` for data loading, parameter fitting, and replay evaluation

---

## Part 4: GitHub + KAUST IBEX Cluster Integration

### 4a. GitHub setup
- Create GitHub repo (or verify existing remote) so code can be pulled on IBEX
- Ensure `.gitignore` covers: `results/`, `data/`, `slurm/slurm_logs/`, `__pycache__/`, `*.pyc`, `*.pdf`
- Push all code changes before cluster runs
- Workflow: develop locally → push to GitHub → pull on IBEX → submit jobs → pull results back

### 4b. Cluster specs (from ../MultiModalContinualGraphLearning reference)
- **Partition:** `batch`
- **GPU:** V100 32GB (`--constraint=v100`)
- **Conda:** `~/miniconda3/bin/conda`, activate via `eval "$(~/miniconda3/bin/conda shell.bash hook)"`
- **Logging:** `slurm/slurm_logs/` with `%x_%J.out` format

### 4c. Slurm file structure (matching graphLearning pattern)

**Runner scripts** (the actual SLURM jobs — each has `#SBATCH` headers):

- **`slurm/run_experiment.sh`** — Generic single-experiment runner
  ```bash
  #!/bin/bash --login
  #SBATCH --partition=batch
  #SBATCH --nodes=1
  #SBATCH --gpus-per-node=1
  #SBATCH --constraint=v100
  #SBATCH --cpus-per-gpu=2
  #SBATCH --mem=32G
  #SBATCH --time=8:00:00
  #SBATCH --output=slurm/slurm_logs/%x_%J.out

  ALGORITHM=${1:?"Algorithm required"}
  NUM_CATEGORIES=${2:?"Num categories required"}
  DECAY_RATE=${3:?"Decay rate required"}
  SEED=${4:?"Seed required"}
  NUM_STUDENTS=${5:-100}
  QUESTIONS=${6:-10000}

  eval "$(~/miniconda3/bin/conda shell.bash hook)"
  conda activate rlucb

  echo "[STARTED] algo=$ALGORITHM K=$NUM_CATEGORIES decay=$DECAY_RATE seed=$SEED"
  python run_experiment.py \
    --algorithm $ALGORITHM \
    --num-categories $NUM_CATEGORIES \
    --decay-rate $DECAY_RATE \
    --seed $SEED \
    --num-students $NUM_STUDENTS \
    --questions $QUESTIONS \
    --output-dir results/
  echo "[SUCCESS] algo=$ALGORITHM K=$NUM_CATEGORIES decay=$DECAY_RATE seed=$SEED"
  ```

- **`slurm/run_ablation.sh`** — Ablation runner (same SBATCH headers, different args)
- **`slurm/run_real_data.sh`** — Real data evaluation runner (longer wall time: 24h, more memory: 64G)

**Submission script** (the one you run to submit everything):

- **`slurm/submit_all.sh`** — Main orchestrator, modeled on graphLearning's `submit_all.sh`
  ```bash
  #!/bin/bash
  set -e
  mkdir -p slurm/slurm_logs results

  SEEDS="42 123 456 789 1024"
  ALGORITHMS="random ucb1 fucb bkt_bandit bkt_thompson thompson epsilon_greedy sw_ucb leitner oracle"
  CATEGORIES="6 20 50 100"
  DECAY_RATES="0.001 0.005 0.01 0.05 0.1"

  COUNT=0

  # ---------------------------------------------------------------
  # Group 1: Main sweep (10 algos × 4 K × 5 λ × 5 seeds = 1000 jobs)
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
  ```

- **`slurm/submit_ablations.sh`** — Ablation-specific submission
- **`slurm/submit_real_data.sh`** — Real data evaluation submission

**Monitoring:**
```bash
# Check queue
squeue -u $USER
# Watch progress
watch -n 30 'grep -h "\[SUCCESS\]\|\[FAILED\]" slurm/slurm_logs/*.out | sort | tail -30'
# Find failures
grep -l "\[FAILED\]" slurm/slurm_logs/*_*.out
```

### 4d. Conda environment
- Create `environment.yml` for reproducible env on cluster
  ```yaml
  name: rlucb
  dependencies:
    - python=3.11
    - numpy
    - scipy
    - pandas
    - matplotlib
    - scikit-learn
    - pip:
      - gymnasium
  ```
- Setup on IBEX: `conda env create -f environment.yml`

---

## Part 5: Paper Writing (LAST — after all experiments and theory are solid)

### New paper structure (modify files in `paper/`)
1. **Abstract** — Reframe: forgetting breaks standard bandits, we fix it
2. **Introduction** — Motivate with crossover phenomenon, state contributions
3. **Problem Formulation** — Formally define the forgetting-bandit problem (NEW)
4. **Algorithms** — F-UCB and BKT-Bandit with intuition (NEW)
5. **Theoretical Analysis** — Regret bounds, lower bound, convergence (NEW)
6. **Related Work** — Expanded: SW-UCB, BKT, spaced repetition, RL-DKT
7. **Experiments** — 10 algorithms, scaling, ablations, real data
8. **Conclusion** — Broader impact on adaptive learning systems

### Key figures needed
- Regret curves (empirical vs. theoretical bound)
- Scaling plots (knowledge vs. categories for each algorithm)
- Forgetting rate sensitivity heatmap
- Cold-start comparison
- Real data replay results
- Ablation bar charts

---

## Part 6: Implementation Sequence (Priority Order)

### Phase 0 — GitHub + project setup
- Verify/create GitHub remote for the RLUCB repo
- Update `.gitignore` (results/, data/, slurm/slurm_logs/, __pycache__/, *.pyc)
- Create `slurm/` directory structure
- Create `environment.yml`
- Create `plans/` directory in project root for all plan files (copy current plan there)
- Push to GitHub so it can be pulled on IBEX
- **Test**: `git remote -v`, verify push succeeds

### Phase A — New selectors (~400 lines)
- Files: `experiment/selectors.py` (add all 8 new selectors)
- Each selector follows the `BaseSelector` interface: `select_category()`, `update()`, `reset()`, `get_statistics()`
- `Student.time_since_exposure` already available — no changes needed for F-UCB
- **Test immediately**: unit tests + smoke test on small config

### Phase B — Simulation generalization (~200 lines)
- Files: `experiment/simulation.py`, `experiment/config.py`
- Generalize to support N algorithm groups (not just 2)
- Add parameter sweep runner
- Add per-student heterogeneity support to `Student`
- **Test**: verify existing UCB vs Random results still reproduce

### Phase C — Slurm integration (~200 lines)
- Create runner scripts: `slurm/run_experiment.sh`, `slurm/run_ablation.sh`, `slurm/run_real_data.sh`
- Create submission scripts: `slurm/submit_all.sh`, `slurm/submit_ablations.sh`, `slurm/submit_real_data.sh`
- Pattern: `submit_*.sh` loops over configs and calls `sbatch -J {name} slurm/run_*.sh {args}`
- Each runner uses positional args (`$1`, `$2`, ...) passed from submit script
- **Test**: dry-run locally (`bash slurm/run_experiment.sh random 6 0.01 42`), then submit test job to IBEX

### Phase D — Run synthetic experiments on cluster
- Submit full parameter sweep (200 configs × 5 seeds = 1000 jobs)
- Collect results, generate visualizations
- Iterate on algorithms if results reveal issues

### Phase E — Real data pipeline (~500 lines)
- New file: `experiment/real_data.py`
- Download scripts, data loaders, MLE parameter fitting
- Replay-based evaluation framework
- **Test**: verify parameter fitting on small data subset

### Phase F — Run real data experiments on cluster
- Submit Duolingo + ASSISTments evaluation jobs
- Collect and visualize results

### Phase G — Theory (pen-and-paper, then LaTeX)
- Regret bound proof for F-UCB (revise multiple times)
- Convergence proof for BKT-Bandit
- Lower bound construction
- **Verify**: empirical regret must fall within theoretical bound curves
- Iterate between theory and experiments until they align

### Phase H — Paper writing (LAST)
- Only after experiments are done and theory is verified
- Rewrite `paper/sec/*.tex` files
- New section files for problem formulation, algorithms, theory
- Update `paper/refs.bib` with ~20-30 new references

---

## Critical Files to Modify

| File | Changes |
|------|---------|
| `experiment/selectors.py` | Add 8 new selector classes |
| `experiment/config.py` | Add algorithm selection, sweep params, heterogeneity |
| `experiment/student.py` | Add per-student parameter variation |
| `experiment/simulation.py` | Generalize to N groups, add sweep runner |
| `experiment/visualization.py` | New plot types for scaling, ablation, regret |
| `experiment/real_data.py` | NEW — data loading, fitting, replay evaluation |
| `run_experiment.py` | Add CLI args for new algorithms and sweeps |
| `slurm/run_experiment.sh` | NEW — SLURM runner for single experiment |
| `slurm/run_ablation.sh` | NEW — SLURM runner for ablation jobs |
| `slurm/run_real_data.sh` | NEW — SLURM runner for real data evaluation |
| `slurm/submit_all.sh` | NEW — orchestrator: submits all sweep jobs |
| `slurm/submit_ablations.sh` | NEW — orchestrator: submits ablation jobs |
| `slurm/submit_real_data.sh` | NEW — orchestrator: submits real data jobs |
| `environment.yml` | NEW — conda env for IBEX cluster |
| `.gitignore` | Update — add results/, data/, slurm_logs/ |
| `plans/` | NEW — all plan files stored here (version-controlled) |

---

## Verification Plan

1. **Unit tests**: Each new selector produces valid category indices, updates state correctly
2. **Smoke test**: Run all 10 algorithms on small config (10 students, 6 categories, 100 questions) — verify no crashes
3. **Reproduction test**: UCB1 vs Random must match existing findings exactly
4. **Sanity checks**: Oracle should always win; Random should be worst or near-worst
5. **Scaling test**: Run with 100 categories — verify memory/time are reasonable
6. **Theory verification**: Empirical regret should fall within theoretical bound curves
7. **Real data pipeline**: Verify parameter fitting produces reasonable α, β, λ values; replay evaluation runs without errors
8. **Statistical rigor**: 5 seeds per config, report mean ± std, Wilcoxon signed-rank tests for significance

---

## Worklog & Session Tracking

- Maintain `WORKLOG.md` at project root — document all changes per session
- Update memory files in `.claude/projects/.../memory/` after each session
- Each worklog entry: date, what was done, what was tested, what's next
- All plan files saved to `plans/` in the project root (version-controlled with the repo)
