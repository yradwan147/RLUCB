# RLUCB NeurIPS Extension — Worklog

## Session 4 — 2026-03-18

### All experiments complete
- K=6: 9/9 ✓ | K=20: 9/9 ✓ | K=50: 8/9 ✓ (one CSV missing) | K=100: 9/9 ✓
- Duolingo: 3/3 ✓ | ASSISTments: 3/3 ✓ (s456 partial)
- Total: 38/39 synthetic + 6/6 real data

### Full Analysis Results

**Algorithm winners by regime:**
- BKT-Bandit wins 7/12 synthetic configs (low-to-medium decay, all K values)
- F-UCB wins 3/12 configs (all high-decay d=0.05)
- Leitner wins 2/12 (K=6 low-decay)

**Quantitative claims for paper:**
1. BKT-Bandit: +8.1% avg knowledge, +5.5% weakest-category over Random (all configs avg)
2. F-UCB: +8.8% avg knowledge over Random, most robust to forgetting rate variation
3. BKT-Bandit converges fastest — overtakes Random by t=80
4. F-UCB dominates high decay: +21-41% over Random at d=0.05
5. Oracle catastrophe: underperforms Random at high K and high decay

**Scaling (K=6→K=100 at d=0.01):**
- All algorithms degrade 77-81% — curse of dimensionality
- F-UCB and BKT-Bandit degrade slightly less than baselines
- SW-UCB least degradation (66%) but poor absolute performance

**Forgetting sensitivity (K=20):**
- F-UCB most robust (range=0.133 across decay rates)
- Random least robust (range=0.197)

**Temporal (K=20 d=0.01):**
- BKT-Bandit overtakes Random at t=80 (very early)
- F-UCB overtakes Random at t=550
- UCB1 never overtakes Random in this config

**Real data:**
- Duolingo (easy, λ≈0): all algorithms ~0.90, Leitner narrowly best. BKT-Bandit +2.9% weakest-cat over Random.
- ASSISTments (hard, λ=0.003): Random wins — adaptive algorithms don't help. Sim-to-real gap.

### Full Code Audit
Ran 3 parallel audits: selectors, simulation framework, real data pipeline.

**Selectors: All 10 correctly implemented** — no algorithm bugs found. Formulas, edge cases, posterior updates all verified.

**Simulation bugs found and fixed:**
1. **CRITICAL: `hash(algo)` for selector seeding** — Python hash() is non-deterministic across runs (PYTHONHASHSEED). Fixed: use `algo_index * 10000` offset instead.

**Real data pipeline bugs found and fixed (explains ASSISTments negative):**
1. **CRITICAL: Timescale mismatch** — decay_rate fitted in per-time_unit (seconds), but replay used per-step. Fixed: replay now uses real deltas from data and time_unit from fitted params.
2. **CRITICAL: Delta discarded in replay** — `real_delta` was loaded but ignored. Fixed: replay advances `current_time` by `real_delta` and computes forgetting in real seconds.
3. **HIGH: Single optimizer initialization** — L-BFGS-B with one start point. Fixed: 5 random restarts.
4. **HIGH: Clipping inside objective** — broke L-BFGS-B gradients at boundaries. Fixed: removed, rely on `bounds` parameter.
5. **MEDIUM: `hash(student_id)` in replay** — non-deterministic. Fixed: use `42 + student_index`.
6. **Added `time_unit` to FittedParams** — stored and used consistently between fit and replay.

### Rerun 3 submitted
- `bash slurm/submit_fixed_real_data.sh` → 7 jobs (3 Duolingo + 3 ASSISTments + 1 missing K=50)
- ASSISTments results should now be meaningful with the fixed pipeline

### What's next (after rerun 3 finishes)
- Transfer and analyze corrected real data results
- Verify ASSISTments now shows adaptive algorithm advantages
- Phase G: Theory (F-UCB regret bounds, BKT-Bandit convergence)
- Phase H: Paper writing (last), have 2/3 seeds for that config

---

## Session 3 — 2026-03-17 (continued)

### Analysis of K=6, K=20, Duolingo
- Identified regime split (BKT-Bandit easy, F-UCB hard)
- Oracle catastrophe finding
- UCB1 exploration trap finding

---

## Session 2 — 2026-03-17

### Bug fixes across 2 reruns
- **Rerun 1**: auto log_frequency → K=20 succeeded, K=50/100 still OOM
- **Rerun 2**: disabled Student.history (root cause: 2×K floats/step/student) → all succeeded

---

## Session 1 — 2026-03-16

### Implementation complete
- 10 algorithms: F-UCB, BKT-Bandit, BKT-Thompson, Thompson, ε-greedy, SW-UCB, Leitner, Oracle + UCB1 + Random
- MultiAlgorithmExperiment, real data pipeline, slurm scripts (chessgcn env, CPU-only)

---

## Paper References (for Phase H)

### Already in `paper/refs.bib`
- Auer et al. 2002 — UCB1 finite-time analysis
- Lattimore & Szepesvári 2020 — Bandit Algorithms textbook
- Ebbinghaus 1885 — Forgetting curve
- Murre & Dros 2015 — Ebbinghaus replication
- Corbett & Anderson 1994 — BKT
- Piech et al. 2015 — Deep Knowledge Tracing (NeurIPS)
- Yudelson et al. 2013 — Individualized BKT
- Clément et al. 2015 — MAB for ITS
- Liu & Koedinger 2014 — Trading off learning with bandits
- Rafferty et al. 2019 — Statistical consequences of MAB in education
- Doroudi et al. 2019 — Where's the reward?
- Settles & Meeder 2016 — Duolingo half-life regression
- Leitner 1972 — Leitner box system
- Pavlik & Anderson 2005 — Practice/forgetting activation model
- VanLehn 2011 — ITS effectiveness review
- Towers et al. 2024 — Gymnasium

### New references to add
- Garivier & Moulines 2011 — SW-UCB
- Besbes et al. 2014 — Non-stationary MAB regret
- Thompson 1933, Chapelle & Li 2011, Agrawal & Goyal 2012, Russo et al. 2018 — Thompson Sampling
- Zhou & Tan 2024 — Variance-dependent regret (NeurIPS 2024)
- He et al. 2024 — Cert-LSVI-UCB (NeurIPS 2024)
- Zhang et al. 2025 — RL-DKT
- Shi et al. 2023 — ALPN
- Reddy et al. 2016 — Optimal spaced repetition (KDD)
- Settles et al. 2018 — SLAM shared task
- Feng et al. 2009 — ASSISTments dataset
- Chaloner & Verdinelli 1995 — Bayesian experimental design
- Lai & Robbins 1985 — Regret lower bounds
- Choffin et al. 2019 — DAS3H (KDD)
