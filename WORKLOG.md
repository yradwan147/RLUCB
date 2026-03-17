# RLUCB NeurIPS Extension — Worklog

## Session 3 — 2026-03-17 (continued)

### Rerun 2 status
- K=50, K=100, ASSISTments jobs were still running at time of transfer
- K=6, K=20, Duolingo fully available — analyzed below

### Full Analysis (K=6, K=20, Duolingo)

**Finding 1: Clear regime split — two novel algorithms each dominate their territory**
- BKT-Bandit wins "easy" regime (low K, low decay): +44% weakest-category over UCB1 at K=6 d=0.01
- F-UCB wins "hard" regime (high K or high decay): +62% avg knowledge over UCB1 at K=6 d=0.05

**Finding 2: Oracle catastrophe at high forgetting**
- Oracle is *worst* algorithm at d=0.05 — greedy weakest-first spreads too thin under rapid decay
- Cautionary result against naive "always quiz the weakest" policies

**Finding 3: UCB1 exploration trap**
- UCB1's confidence bounds ignore forgetting → wastes time on decayed arms
- BKT-Bandit degrades least with K scaling (-67% vs Oracle's -76% going K=6→K=20)

**Finding 4: BKT-Bandit = equity maximizer**
- Best weakest-category knowledge in tractable regimes consistently
- +38% over Random on weakest skill at K=6 d=0.01

**Finding 5: Duolingo real data validation**
- BKT-Bandit +2.9% weakest-category over Random (λ≈0, "easy" regime)
- F-UCB ≈ UCB1 as expected with near-zero forgetting
- Consistent across all 3 seeds (std < 0.004)

**Statistical robustness**: Cross-seed std = 0.001–0.004 across all configs.

### Pending
- K=50, K=100 results (still running on IBEX) — will show scaling behavior
- ASSISTments results (still running) — different domain, likely higher forgetting
- Transfer and analyze when complete

### Paper narrative (draft)
*"Standard bandits fail in educational settings because they ignore forgetting. We formalize why (Oracle catastrophe, UCB1 exploration trap), propose two complementary solutions (F-UCB for hard regimes, BKT-Bandit for easy regimes), prove guarantees, and validate on real data."*

---

## Session 2 — 2026-03-17

### Results from Run 1
- K=6 sweep: 9/9 succeeded
- K≥20 sweep: 27/27 OOM killed (exit 137)
- Duolingo: 3/3 succeeded
- ASSISTments: 6/6 failed (timestamp parsing)

### Bugs Fixed
- **Rerun 1**: auto log_frequency, 32G memory → K=20 succeeded, K=50/100 still OOM
- **Rerun 2**: disabled Student.history (root cause: 2×K floats per step per student), robust ASSISTments timestamp parsing

---

## Session 1 — 2026-03-16

### What was done
- Phases 0, A, B, C, E complete
- 10 algorithms: F-UCB, BKT-Bandit, BKT-Thompson, Thompson, ε-greedy, SW-UCB, Leitner, Oracle + UCB1 + Random
- MultiAlgorithmExperiment, real data pipeline, slurm scripts
- 42 initial jobs submitted

---

## Paper References (for Phase H)

### Already in `paper/refs.bib`
- Auer et al. 2002 — UCB1 finite-time analysis
- Lattimore & Szepesvári 2020 — Bandit Algorithms textbook
- Ebbinghaus 1885 — Forgetting curve
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

### New references to add

**Non-stationary bandits (F-UCB theory):**
- Garivier & Moulines 2011 — SW-UCB for switching bandits
- Besbes et al. 2014 — Non-stationary MAB regret bounds
- Russac et al. 2019 — D-LinUCB for non-stationary environments

**Thompson Sampling:**
- Thompson 1933 — Original TS paper
- Chapelle & Li 2011 — Empirical evaluation of TS
- Agrawal & Goyal 2012 — TS regret bounds
- Russo et al. 2018 — Tutorial on Thompson Sampling

**NeurIPS 2024 context:**
- Zhou & Tan 2024 — Variance-dependent regret in contextual bandits
- He et al. 2024 — Cert-LSVI-UCB constant regret bounds

**Knowledge tracing + RL:**
- Zhang et al. 2025 — RL-DKT (Nature Scientific Reports)
- Shi et al. 2023 — ALPN: AKT + Entropy-enhanced PPO
- Ghosh et al. 2020 — Contextual bandits with latent confounders

**Spaced repetition:**
- Pimsleur 1967 — Graduated interval recall
- Reddy et al. 2016 — Optimal scheduling for spaced repetition (KDD)

**Educational data mining:**
- Settles et al. 2018 — SLAM shared task (Duolingo dataset)
- Feng et al. 2009 — ASSISTments dataset
- Baker & Inventado 2014 — EDM survey

**Bayesian experimental design (BKT-Bandit theory):**
- Chaloner & Verdinelli 1995 — Bayesian experimental design review
- Lindley 1956 — Information gain measure

**Regret lower bounds:**
- Lai & Robbins 1985 — Asymptotically efficient allocation

**Real data baselines:**
- Choffin et al. 2019 — DAS3H: skills + forgetting (KDD)
- Wilson et al. 2016 — Bayesian extensions of IRT
