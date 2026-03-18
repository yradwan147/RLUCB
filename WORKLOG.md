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

### Testing (in progress)
- Running comparison of all 9+2 algorithms on K=6 d=0.01

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
