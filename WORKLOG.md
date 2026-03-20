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

### IBEX run submitted
`bash slurm/submit_meta_v2.sh` → 42 jobs (36 synthetic + 6 real data, all 14 algorithms)
Waiting for results to validate at scale before further iteration.

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
