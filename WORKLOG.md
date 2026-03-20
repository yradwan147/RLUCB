# RLUCB NeurIPS Extension — Worklog

## Session 7 — 2026-03-19

### Full IBEX Results: 14 Algorithms (42 jobs, all succeeded)

**MetaSelector v1 at scale (100 students, 10K questions):**
- Top-3 in 9/12 configs (75%) — matches BKT-Bandit and F-UCB consistency
- **Never wins #1** — always close second/third
- Average gap to best: 4.7%
- **Failure mode**: K=20 d=0.005 — rank 12th, 15% gap

**Root cause diagnosed**: correctness-based UCB reward favors F-UCB (high accuracy from revisiting) over BKT-Bandit (targets weak categories, lower accuracy but higher knowledge). Confirmed: Oracle has lowest accuracy (0.527) but highest knowledge (0.581).

### MetaSelector v2: Blended ΔK + Correctness Reward

**Research**: Investigated off-policy evaluation, reward shaping, dueling bandits, Elo ratings, BKT observation models, EWMA knowledge estimation, CORRAL algorithm.

**Solution**: EWMA-based per-category knowledge estimation. Reward = blend of ΔK (knowledge gain) and correctness, shifting from ΔK-dominated early to accuracy-dominated late:
```
reward = (1 - t/1000) * ΔK * 100 + (t/1000) * (accuracy - 0.5)
```

**4 iterations:**
1. Pure ΔK → great at low decay (#2), terrible at high decay (33.7% gap)
2. ΔK without forgetting decay → #1 at K=6 d=0.005, still bad at d=0.05
3. Blended ΔK + correctness → **best of both worlds**
4. Also fixed run_real_data.py to use all 14 algorithms from registry

**MetaSelector v2 local results (30 students, 2000q):**
| Config | Rank | Gap | vs v1 |
|--------|------|-----|-------|
| K=6 d=0.005 | 4th | 0.8% | same |
| K=6 d=0.01 | 4th | 2.4% | same |
| K=6 d=0.05 | **#1** | 0% | same |
| K=20 d=0.005 | 4th | 4.6% | **was 12th!** |
| K=20 d=0.01 | **3rd** | 6.7% | was 4th |
| K=20 d=0.05 | **#2** | 1.1% | same |

### MetaSelector v2 IBEX Results (42 jobs, all succeeded)
- 8/12 top-3 (was 9/12 in v1), never #1
- Fixed K=20 d=0.005 failure but regressed K=6 d=0.01 to 8th
- Reliably #2 at d=0.05 (<1% gap to F-UCB)
- Equity: poor (rank ~8/14)

### Reward Signal Iterations (6 attempts)
1. Correctness rate (v1): biased toward F-UCB
2. Blended ΔK + correctness (v2): helped high-decay, hurt low-decay
3. Pure ΔK: noisy signal (all rewards ≈ -0.002, barely distinguishable)
4. Tenure-based: too coarse (50-step blocks), rank 4 everywhere
5. Category-scoring v3: recreates BKT-Bandit over expert candidates
6. Category-scoring v4 + urgency: no improvement

### Key Diagnosis from Trajectory Analysis
- MetaSelector locks onto F-UCB early because F-UCB gets higher correctness
- At K=20 d=0.005, meta tracks F-UCB (score 0.26) while BKT-Bandit is at 0.31
- ΔK rewards are all ≈ -0.002 to +0.0002 — too small to distinguish experts
- Correctness is INVERSELY correlated with knowledge-targeting at short timescales

### Insights from Orabona's Online Learning Textbook (Ch 10-11)
- **Ch 10.5**: Combining two OCO algorithms gives min(Regret_A, Regret_B) + O(1)
- **Ch 10.6**: Reduction to LEA via coin-betting — parameter-free, optimal
- **Ch 11 EXP3**: Importance-weighted loss estimator is unbiased — need correct loss definition first
- **Key**: Define loss as (1 - knowledge_improvement), not (1 - correctness), then apply EXP3

### MetaSelector v3: Unified Scoring (No Expert Tracking)

**Key insight**: Stop tracking experts entirely. Fuse BKT-Bandit's posterior with F-UCB's three-term structure into a single category scorer. Experts only provide candidate filtering.

**Audited against Orabona's textbook**: Valid UCB with non-stationary confidence width (§6.10 Optimistic OMD). Expert candidates reduce action set K→M (§6.8 LEA theory).

**Formula**: `score = (1-mean)*exp(-λt) + 0.5*(1-exp(-λt)) + c*std`
- Term 1: time-decayed weakness (old observations lose weight)
- Term 2: additive forgetting urgency (like F-UCB)
- Term 3: posterior uncertainty exploration (like BKT-Bandit)

**Iterations to get here:**
1. `(1-mean) + c*std*(1+λt)` — good at low decay, bad at high (urgency multiplied small std)
2. `exp(λt)` urgency — #1 at K=6 d=0.01 but still bad at d=0.05
3. Additive urgency — still too weak
4. **F-UCB three-term structure** — top-3 in 5/6 configs!

**Local results (30 students, 2000q):**
| Config | Meta Rank | Gap |
|--------|-----------|-----|
| K=6 d=0.005 | **#3** | 0.0% |
| K=6 d=0.01 | **#3** | 0.3% |
| K=6 d=0.05 | **#3** | 27.6% |
| K=20 d=0.005 | 4th | 7.2% |
| K=20 d=0.01 | **#3** | 9.3% |
| K=20 d=0.05 | **#3** | 12.6% |

Avg rank 3.2, top-3 in 5/6. Best consistency yet.

### IBEX submission ready
`bash slurm/submit_meta_v3.sh` → 42 jobs

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
