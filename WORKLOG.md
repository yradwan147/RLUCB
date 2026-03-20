# RLUCB NeurIPS Extension — Worklog

## Session 7 — 2026-03-19

### Full IBEX Results: 14 Algorithms (42 jobs, all succeeded)

**MetaSelector at scale (100 students, 10K questions):**
- Top-3 in 9/12 configs (75%) — matches BKT-Bandit and F-UCB consistency
- **Never wins #1** — always close second/third
- Average gap to best: 4.7%
- #2 at all d=0.05 configs (closely tracks F-UCB)
- **Failure mode**: K=20 d=0.005 — rank 12th, 15% gap

**Root cause diagnosed**: correctness-based UCB reward favors F-UCB (high accuracy from revisiting) over BKT-Bandit (targets weak categories, lower accuracy but higher knowledge). The meta-learner picks the wrong expert in BKT-Bandit-optimal regimes.

**Equity weakness**: MetaSelector rank 7.5/13 for weakest-category. adaptive_whittle dominates equity (9/12 top-3).

**Consistency rankings (top-3 out of 12 configs):**
- BKT-Bandit: 9/12 (wins 5 outright)
- F-UCB: 9/12 (wins 4 outright)
- MetaSelector: 9/12 (never #1)
- adaptive_whittle: best equity (9/12 top-3 weakest-cat)

**Real data**: Still only 10 algorithms (run_real_data.py default list doesn't include Whittle/meta). Not critical — synthetic results are the main story.

### Problem to solve next
Need a reward signal for MetaSelector that correctly identifies the best expert even when the best expert has LOWER accuracy (because it targets weak categories). Current signal: correctness rate → biased toward F-UCB. Need: knowledge-aware signal.

---

## Session 5-6 — 2026-03-18

### PD-RMAB Direction Pivot
- Implemented Whittle index computation (advantage-based, budget-aware)
- 5 iterations of Whittle selectors: whittle, pd_whittle, adaptive_whittle
- Key insight: Whittle advantage peaks at k≈0.3-0.5 (zone of proximal development)

### MetaSelector Implementation
- 4 iterations: EXP4 → follow-leader → knowledge-gain → UCB-over-experts
- UCB-over-experts best: #1 at d=0.05 locally, #2 at scale

### Session 4 — Full code audit, bug fixes
### Session 2-3 — Bug fixes, K=6/K=20 analysis
### Session 1 — 10 algorithms, framework, real data pipeline

---

## Paper References (see previous entries for full list)
