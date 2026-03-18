# RLUCB NeurIPS Extension — Worklog

## Session 6 — 2026-03-18

### Full IBEX Results: Synthetic (36 jobs, 13 algorithms, 100 students, 10K questions)

All 36 jobs succeeded. 13 algorithms × 4 K values × 3 decay rates × 3 seeds.

**Overall rankings (averaged across all 12 configs):**
1. BKT-Bandit (0.2444)
2. F-UCB (0.2421)
3. adaptive_whittle (0.2406)
4. pd_whittle (0.2393)
5. whittle (0.2370)
6. random (0.2367)
7. thompson (0.2310)
...
11. ucb1 (0.2182)

**Clean three-way regime partition:**
- BKT-Bandit: wins K≥20, d≤0.01 (5/12 configs)
- F-UCB: wins d=0.05 regardless of K (4/12 configs)
- Whittle variants: win K=6, d≤0.01 (3/12 configs)

**Whittle variant highlights:**
- Best for equity: PD-Whittle #1 in weakest-category (+11.5% over random)
- Beat UCB1 by 5-9% across all configs
- Fall below random at K≥50 (advantage computation doesn't scale)

**Surprising findings:**
- BKT-Bandit beats Oracle in 10/12 configs (Oracle is greedy, not optimal)
- Random beats UCB1 overall (exploration trap is severe)
- ε-greedy quietly top-3 in 8/12 configs

**Missing: Real data with Whittle variants** — previous real data runs had only 10 algorithms. Created `slurm/submit_whittle_real_data.sh` for 6 more jobs.

---

## Session 5 — 2026-03-18

### Direction Pivot: Parametric-Decay Restless Bandits (PD-RMAB)

**Research findings**: Problem is formally an RMAB. Existing algorithms treat transitions as general unknown matrices. We exploit parametric forgetting structure.

**5 algorithm iterations:**
1. Raw Whittle → terrible (one-step approx peaks at k≈0.5)
2. Advantage-based (V_active - V_passive) → correctly favors weak
3. Normalized [0,1] + exploration → competitive
4. Budget-aware (1/K active fraction) → improved K=20
5. AdaptiveWhittle (Whittle + urgency blend) → beats Oracle at K=6 d=0.005

**Key theoretical insight**: Whittle advantage peaks at k≈0.3-0.5, not k=0. This explains Oracle catastrophe — quizzing very weak arms is suboptimal (zone of proximal development).

---

## Session 4 — 2026-03-18

- Full code audit: all 10 selectors correct
- Fixed: hash seeding, real data timescale mismatch, ASSISTments timestamp parsing
- Real data results: Duolingo (easy, λ≈0), ASSISTments (hard, λ=0.003)

## Session 2-3 — 2026-03-17

- Bug fixes: OOM (disabled Student.history), auto log_frequency
- K=6,20 analysis, Duolingo results

## Session 1 — 2026-03-16

- Implemented 10 algorithms, MultiAlgorithmExperiment, real data pipeline, slurm scripts

---

## Paper Narrative (Final)

**Title**: "Restless Bandits for Adaptive Learning: Principled Question Selection with Forgetting Dynamics"

**Story**:
1. Adaptive question selection with forgetting is a restless bandit problem
2. Standard UCB ignores decay → exploration trap, worse than random at scale
3. We propose three novel approaches: Whittle advantage-based, PD-Whittle (Bayesian + advantage), AdaptiveWhittle (urgency blend)
4. BKT-Bandit wins at medium-high K, F-UCB wins at high decay, Whittle wins at low decay + best equity
5. **No single algorithm dominates** — regime-dependent optimality is the key insight
6. Whittle advantage analysis theoretically explains Oracle catastrophe (zone of proximal development)
7. PD-Whittle achieves best equity (weakest-category) across all configs

---

## Paper References

### Already in refs.bib (16 refs)
Auer 2002, Lattimore 2020, Ebbinghaus 1885, Murre 2015, Corbett 1994, Piech 2015, Yudelson 2013, Clément 2015, Liu 2014, Rafferty 2019, Doroudi 2019, Settles 2016, Leitner 1972, Pavlik 2005, VanLehn 2011, Towers 2024

### New references to add (~25)
**RMAB/Whittle**: Whittle 1988, Niño-Mora 2023 (review), NeurWIN (NeurIPS 2021), Neural-Q-Whittle (NeurIPS 2023), Index-aware RL (NeurIPS 2022), EduQate (AAMAS 2025)
**Non-stationary bandits**: Garivier & Moulines 2011, Besbes 2014
**Thompson Sampling**: Thompson 1933, Chapelle & Li 2011, Agrawal & Goyal 2012, Russo 2018
**NeurIPS 2024**: Zhou & Tan (variance-dependent), He et al. (Cert-LSVI-UCB)
**Knowledge tracing + RL**: Zhang 2025 (RL-DKT), Shi 2023 (ALPN)
**Spaced repetition**: Tabibian 2019 (PNAS), Reddy 2016 (KDD)
**Datasets**: Settles 2018 (SLAM), Feng 2009 (ASSISTments)
**Bayesian design**: Chaloner & Verdinelli 1995
**Lower bounds**: Lai & Robbins 1985
**Real data**: Choffin 2019 (DAS3H)
