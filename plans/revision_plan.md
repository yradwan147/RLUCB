# Plan: Comprehensive Paper Revision Based on Stanford Agentic Review

## Context

The Stanford AI reviewer gave a "lean positive, contingent on clarifications and additions" assessment. The review identifies 7 specific weaknesses and 7 questions. The overall verdict: strong empirical contribution with the Oracle catastrophe insight, but needs (1) missing baselines, (2) clearer terminology, (3) sensitivity analyses, (4) more seeds, (5) additional metrics, (6) better real-data evaluation description, and (7) theoretical justification.

This plan categorizes every issue by effort level and proposes a concrete action for each.

---

## Category A: Paper-Only Fixes (No New Experiments)

### A1. Rename "Oracle" to "Myopic-Greedy" throughout
**Issue:** Reviewer says "Oracle" is misleading — it's argmin(knowledge), not DP-optimal.
**Fix:** Rename to "Greedy-Min" or "Myopic-Min-K" in all .tex files. Add footnote explaining it uses perfect current-state knowledge but is myopic (no planning). Clarify it's an upper bound on greedy policies, not global optimality.
**Files:** `sec/3_method.tex`, `sec/4_experiments.tex`, `sec/1_intro.tex`, `sec/0_abstract.tex`, all tables

### A2. Soften "first formalization as RMAB" claim
**Issue:** Prior work (Reddy 2016, Tabibian 2019, EduQate 2024) has used RMAB/optimal-control for education.
**Fix:** Change to "we extend RMAB-based approaches to a comprehensive quizzing benchmark with Ebbinghaus forgetting and exploration." Add citations to Reddy, Tabibian, EduQate in related work.
**Files:** `sec/0_abstract.tex`, `sec/1_intro.tex`, `sec/2_related.tex`

### A3. Add missing related work citations
**Issue:** Missing discounted Thompson Sampling (dTS/dOTS), EXP4/Corral, Bayesian nonparametrics.
**Fix:** Add paragraph in related work on discounted/non-stationary TS (cite Raj & Kalyani 2017 arXiv:1707.09727). Add brief positioning of MetaSelector vs EXP4/Corral (Agarwal 2017). Cite DPPS and variance-adaptive bandits.
**Files:** `sec/2_related.tex`, `refs.bib`

### A4. Clarify MetaSelector theory claims
**Issue:** Paper implies EXP4/Corral connection but implementation is just "unified UCB scoring over expert-filtered actions."
**Fix:** Describe it as "expert-guided UCB with forgetting-aware confidence" rather than claiming formal EXP4. Add brief discussion of why pure EXP4 failed (reward signal mismatch, see worklog iterations).
**Files:** `sec/3_method.tex`

### A5. Clarify real-data replay protocol
**Issue:** Reviewer asks how outcomes are obtained when bandit picks unlogged actions.
**Fix:** Add explicit paragraph: "When the bandit selects a skill with available logged data, we use the real outcome. When no logged interaction exists, we simulate the outcome using the fitted model P(correct) = k. This is a model-based evaluation, not pure off-policy estimation." Acknowledge limitation. Reference IPS/DR as future work.
**Files:** `sec/4_experiments.tex`, `sec/supplementary.tex`

### A6. Add Whittle implementation details to supplementary
**Issue:** Discretization granularity, convergence, runtime not reported.
**Fix:** Add table: num_states=50, convergence threshold 1e-8, 80 binary search × 200 VI steps, precomputation ~2s per config, O(1) lookup at runtime. Add sensitivity note.
**Files:** `sec/supplementary.tex`

### A7. Provide Bayesian justification for BKT-Bandit posterior decay
**Issue:** "No theoretical analysis — regret bounds or calibration guarantees."
**Fix:** Add paragraph in methodology connecting the exponential posterior shrinkage to discounted posteriors (Raj & Kalyani 2017) and hazard-model interpretation: "The posterior decay can be interpreted as a discount factor on past observations, equivalent to a geometric forgetting window of effective size 1/λ."
**Files:** `sec/3_method.tex`

---

## Category B: New Experiments Required (Code + IBEX)

### B1. Add Discounted Thompson Sampling (dTS) baseline — CRITICAL
**Issue:** "Its omission as a baseline is a notable gap."
**Action:** Implement `DiscountedTSSelector` in `selectors.py`. Uses discounted posterior: α_t = 1 + γ·Σ_{s<t} γ^{t-s} · correct_s. Select via Thompson Sampling from discounted Beta. γ = exp(-λ) uses the decay rate.
**Effort:** ~60 lines of code + IBEX run
**Files:** `experiment/selectors.py`

### B2. λ mis-specification sensitivity analysis — CRITICAL
**Issue:** "How sensitive are F-UCB and BKT-Bandit to λ mis-specification?"
**Action:** Run experiments where algorithm uses λ̂ ∈ {0.5λ, 0.8λ, λ, 1.2λ, 2λ} while true λ stays fixed. Test on K=6 d=0.01 (3 seeds). Report performance degradation vs Random.
**Effort:** Modify config to support separate algorithm_decay_rate, run ~15 jobs
**Files:** `experiment/selectors.py` (pass separate λ), `experiment/config.py`, new slurm script

### B3. Increase seeds from 3 to 10
**Issue:** "3 seeds limits statistical power; 10+ seeds would be preferable."
**Action:** Rerun key configs (K=6 and K=20, all 3 decay rates = 6 configs) with seeds 42,123,456,789,1024,1337,2024,3141,4242,5555. Report mean ± std and paired t-tests.
**Effort:** 6 configs × 1 job each = 6 IBEX jobs (each runs all algorithms)
**Files:** New slurm script

### B4. Additional metrics: AUC and time-to-mastery
**Issue:** "Reporting only final knowledge obscures early/mid-horizon dynamics."
**Action:** Compute from existing CSVs (no new experiments needed):
- **AUC**: area under average_knowledge curve (trapezoid rule)
- **Time-to-mastery**: first timestep where weakest_category_avg ≥ 0.5 (or 0.3 for high-decay)
- **Post-gap retention**: knowledge at t=10000 after a simulated 500-step gap
Report as supplementary table.
**Effort:** Python analysis script only
**Files:** Analysis script, `sec/supplementary.tex`

### B5. Whittle discretization sensitivity (supplementary)
**Issue:** "How sensitive are results to the discretization grid?"
**Action:** Run K=6 d=0.01 with num_states ∈ {20, 50, 100, 200} and compare Whittle selector performance. Quick local test, no IBEX needed.
**Effort:** ~10 minutes local computation
**Files:** `sec/supplementary.tex`

---

## Category C: Nice-to-Have (Lower Priority)

### C1. Add limited-lookahead Oracle
**Issue:** "Compare against a limited-lookahead or approximate DP policy."
**Action:** Implement a 2-step lookahead Oracle that considers both immediate weakness and expected knowledge after quizzing. Would strengthen the "Oracle catastrophe" claim.
**Effort:** ~100 lines + local testing. Not strictly required but strengthens the paper.

### C2. Exposure distribution / Gini coefficient analysis
**Issue:** Related to equity discussion.
**Action:** Compute Gini coefficient of exposure distribution for each algorithm. Some already target weak categories more (BKT-Bandit), others spread evenly (Random).
**Effort:** Analysis script only.

### C3. F-UCB urgency weight (γ) sensitivity
**Issue:** "No analysis of optimal weighting."
**Action:** Test γ ∈ {0.1, 0.3, 0.5, 0.7, 1.0} on K=6 d=0.01. Report in supplementary.
**Effort:** Quick local test.

---

## Implementation Order

1. **Paper fixes (A1-A7)** — Can be done immediately, no experiments
2. **dTS baseline (B1)** — Implement + add to registry
3. **λ sensitivity (B2)** — Requires code modification for separate algorithm λ
4. **Additional metrics (B4)** — Compute from existing CSVs
5. **Whittle sensitivity (B5)** — Quick local test
6. **More seeds (B3)** — IBEX jobs (longest wall-clock time)
7. **Submit IBEX jobs** for B1, B2, B3 together (~25 total jobs)
8. **Rewrite paper sections** with new results
9. **Recompile and re-review**

---

## IBEX Job Estimate

| Task | Jobs | Notes |
|------|------|-------|
| dTS baseline (B1) | 6 | K={6,20} × 3 decay × 1 seed (quick validation) |
| λ sensitivity (B2) | 5 | K=6 d=0.01 × 5 λ̂ values × 1 seed |
| More seeds (B3) | 6 | K={6,20} × 3 decay × 7 new seeds (batch) |
| **Total** | **~17** | Well within 50-job limit |

---

## Verification

After all changes:
1. dTS baseline appears in all tables and figures
2. "Oracle" renamed to "Myopic-Greedy" everywhere
3. λ sensitivity figure/table in supplementary
4. AUC and time-to-mastery tables in supplementary
5. 10 seeds with mean ± std reported for key configs
6. Real-data replay protocol explicitly described
7. Whittle details (discretization, runtime) in supplementary
8. Paper recompiles cleanly
9. Re-submit to Stanford reviewer for score improvement

---

## Critical Files to Modify

| File | Changes |
|------|---------|
| `experiment/selectors.py` | Add DiscountedTSSelector, update registry |
| `experiment/config.py` | (possibly) add algorithm_decay_rate for sensitivity |
| `paper/sec/0_abstract.tex` | Soften claims, rename Oracle |
| `paper/sec/1_intro.tex` | Soften claims, rename Oracle |
| `paper/sec/2_related.tex` | Add dTS, EXP4/Corral, prior RMAB citations |
| `paper/sec/3_method.tex` | Rename Oracle, add BKT justification, clarify MetaSelector |
| `paper/sec/4_experiments.tex` | Rename Oracle, add replay clarification, new metrics |
| `paper/sec/supplementary.tex` | Whittle details, λ sensitivity, AUC/mastery tables |
| `paper/refs.bib` | Add ~5 new references |
| `slurm/submit_revision.sh` | New script for all revision experiments |
