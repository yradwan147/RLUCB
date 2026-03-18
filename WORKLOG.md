# RLUCB NeurIPS Extension — Worklog

## Session 5 — 2026-03-18 (continued)

### Direction Pivot: Parametric-Decay Restless Bandits (PD-RMAB)

**Problem**: Previous results too weak for NeurIPS. F-UCB/BKT-Bandit are heuristic modifications of UCB — marginal improvements, no algorithmic novelty.

**Research findings**:
- Our problem is formally a **Restless Multi-Armed Bandit (RMAB)** where arms decay when not pulled
- Existing RMAB learning algorithms treat transitions as general unknown matrices (O(S²) params)
- In education, forgetting follows **known parametric forms** with only 2-3 params
- **Nobody has exploited this structure** — this is the genuine novelty

**Accepted NeurIPS papers in this space require**:
- New algorithms with provable guarantees (not just "applied X to Y")
- Neural-Q-Whittle (NeurIPS 2023): finite-time convergence for neural Whittle
- Index-aware RL (NeurIPS 2022): polynomial regret via index structure
- EduQate (AAMAS 2025): network structure in RMAB for education (but still general learning)

**Our contribution: PD-RMAB**
- Exploits parametric forgetting structure → O(d√T) regret instead of O(S²√T)
- Closed-form Whittle index for education-forgetting model
- Indexability proof
- Connection to cognitive science spacing predictions

### Implementation Progress
- Created `experiment/whittle.py`:
  - `build_transition_matrices()` — discretized MDP for single arm
  - `compute_whittle_indices()` — binary search + value iteration
  - `compute_whittle_index_table()` — full lookup table
  - `approximate_whittle_index()` — closed-form one-step approximation

- Added to `experiment/selectors.py`:
  - `WhittleIndexSelector` — uses precomputed exact indices with estimated knowledge
  - `PDWhittleSelector` — online learning version with Bayesian decay estimation + decay-aware exploration bonus
  - Both registered in SELECTOR_REGISTRY

**Key finding from Whittle index computation:**
- Index peaks at k≈0.3-0.5, NOT at k=0 (weakest)
- This explains Oracle catastrophe: "always quiz weakest" is suboptimal!
- At very low k, quizzing helps little (student mostly wrong, learning gain small)
- The optimal policy quizzes the "zone of proximal development" (k≈0.3-0.5)

### Algorithm Iteration Results

**Iteration 1** — Raw Whittle index: terrible (worst at all decay rates)
- Problem: one-step approximation peaks at k≈0.5, doesn't favor weak categories

**Iteration 2** — Advantage-based index (V_active - V_passive):
- Much better! Monotonically decreasing with k (correctly favors weak)
- But exploration bonus negligible (W~40 vs bonus~0.1)

**Iteration 3** — Normalized indices [0,1] + exploration:
- WhittleIndex now 5th→5th→3rd across decay rates
- PD-Whittle: 4th, 3rd, 3rd — competitive at low/medium decay

**Iteration 4** — Budget-aware advantage (active_fraction = 1/K):
- K=20 d=0.005: PD-Whittle **2nd** (ahead of BKT-Bandit!)
- Still behind F-UCB at high decay

**Current best results (30 students, 2000 questions):**
| Config | #1 | PD-Whittle rank |
|--------|----|----|
| K=6 d=0.005 | Oracle | 4th (0.787, gap <1%) |
| K=6 d=0.01 | Oracle | ~3rd |
| K=6 d=0.05 | F-UCB | 5th |
| K=20 d=0.005 | Random | **2nd** |
| K=20 d=0.01 | BKT-Bandit | 4th |
| K=20 d=0.05 | F-UCB | 5th |

### Plan saved to: `plans/pd_rmab_plan.md`

---

## Previous Sessions

### Session 4 — 2026-03-18
- All experiments complete (38/39 synthetic + 6/6 real data)
- Full code audit: all 10 selectors correct
- Fixed: hash seeding, real data timescale mismatch, ASSISTments timestamp parsing
- Key findings: BKT-Bandit wins 7/12 configs, F-UCB wins 3/12, Oracle catastrophe confirmed on real data

### Session 2-3 — 2026-03-17
- Bug fixes: OOM (disabled Student.history), auto log_frequency
- Analysis of K=6, K=20, Duolingo results

### Session 1 — 2026-03-16
- Implemented 10 algorithms, MultiAlgorithmExperiment, real data pipeline, slurm scripts
