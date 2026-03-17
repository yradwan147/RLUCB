# RLUCB NeurIPS Extension — Worklog

## Session 2 — 2026-03-17

### Results from Run 1
- **K=6 sweep: 9/9 succeeded** (all 10 algos × 3 decay rates × 3 seeds)
- **K≥20 sweep: 27/27 failed** — OOM killed (exit 137), log_frequency=1 too memory-heavy
- **Duolingo real data: 3/3 succeeded** — fitted params + replay + fitted sim
- **ASSISTments real data: 6/6 failed** — datetime string timestamp parsing bug

### Key Findings (K=6, 10K questions, 100 students)
- **F-UCB dominates at high forgetting (d=0.05)**: avg_k=0.211, beats Oracle (0.122) by 73%, beats UCB1 (0.130) by 62%
- **BKT-Bandit wins at medium forgetting (d=0.01)**: avg_k=0.581, matches Oracle (0.584) within 0.6%
- **Leitner wins at low forgetting (d=0.005)**: avg_k=0.787, near-Oracle (0.790)
- **Oracle paradox at d=0.05**: worst performer — spreads too thin under rapid decay
- **F-UCB advantage grows with forgetting rate** — the harder the problem, the more it helps
- **BKT-Bandit best weakest-category at d=0.01** (0.522 vs UCB1's 0.468)

### Duolingo Real Data (500 students)
- Fitted params: α=0.126, β=0.087, λ≈0 (near-zero forgetting), k0=0.91
- Results consistent across 3 seeds
- BKT-Bandit best weakest-category among non-oracle (0.774 vs UCB1's 0.753)
- F-UCB ≈ UCB1 (as expected with λ≈0)

### Bugs Fixed (rerun 1 — K=20 now works)
1. **OOM fix #1**: auto log_frequency = max(1, questions // 1000), caps at ~1000 data points
2. **ASSISTments fix attempt**: datetime conversion (didn't trigger — dtype check fragile)
3. **Memory**: bumped slurm to 32G

### Rerun 1 results
- K=20: 9/9 succeeded
- K=50: 9/9 OOM (exit 137) — Student.history was the culprit
- K=100: 9/9 OOM — same
- ASSISTments: 3/3 failed — timestamp fix didn't trigger

### Bugs Fixed (rerun 2)
1. **OOM fix #2**: `Student.track_history=False` by default. Was storing `knowledge_before` + `knowledge_after` (2×K floats) per step per student. For K=100: ~16GB total.
2. **ASSISTments fix #2**: use `is_numeric_dtype()` instead of `dtype == object`

### Re-run 2 submitted
- `bash slurm/submit_failed_rerun2.sh` → 21 jobs (18 K=50,100 + 3 ASSISTments)

### What's next
- Collect rerun 2 results (should complete K=6,20,50,100 + Duolingo + ASSISTments)
- Generate publication-quality visualizations across all K values
- Phase G: Theoretical analysis (F-UCB regret bounds)
- Phase H: Paper writing

---

## Session 1 — 2026-03-16

### What was done
- Phases 0, A, B, C, E complete (see below)
- 42 jobs submitted on IBEX

**Phase A** — 8 new selectors: F-UCB, BKT-Bandit, BKT-Thompson, Thompson, ε-greedy, SW-UCB, Leitner, Oracle
**Phase B** — MultiAlgorithmExperiment for N algorithm groups
**Phase C** — Slurm integration (CPU-only, chessgcn env)
**Phase E** — Real data pipeline (Duolingo + ASSISTments)

---

## Paper References (for Phase H)

### Already in `paper/refs.bib`
- Auer et al. 2002 — UCB1 finite-time analysis (foundational)
- Lattimore & Szepesvári 2020 — Bandit Algorithms textbook
- Ebbinghaus 1885 — Forgetting curve (foundational)
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

### New references to add for extended paper

**Non-stationary bandits (theory for F-UCB):**
- Garivier & Moulines 2011 — "On Upper-Confidence Bound Policies for Switching Bandit Problems" (SW-UCB, foundation for our non-stationarity argument)
- Besbes et al. 2014 — "Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards" (variation budget regret bounds)
- Russac et al. 2019 — "Weighted Linear Bandits for Non-Stationary Environments" (D-LinUCB)

**Thompson Sampling:**
- Thompson 1933 — "On the likelihood that one unknown probability exceeds another" (original)
- Chapelle & Li 2011 — "An Empirical Evaluation of Thompson Sampling" (empirical comparison vs UCB)
- Agrawal & Goyal 2012 — "Analysis of Thompson Sampling for the Multi-Armed Bandit Problem" (regret bounds for TS)
- Russo et al. 2018 — "A Tutorial on Thompson Sampling" (comprehensive survey)

**Variance-dependent / contextual bandits (NeurIPS 2024):**
- Zhou & Tan 2024 — "How Does Variance Shape the Regret in Contextual Bandits?" (NeurIPS 2024)
- He et al. 2024 — "Cert-LSVI-UCB: Constant instance-dependent regret bounds in RL" (NeurIPS 2024)

**Knowledge tracing + RL (competing methods):**
- Zhang et al. 2025 — "Integrating RL with Dynamic Knowledge Tracing (RL-DKT)" (Nature Scientific Reports)
- Shi et al. 2023 — "Adaptive Learning Path Navigation (ALPN): AKT + Entropy-enhanced PPO" (arXiv)
- Ghosh et al. 2020 — "Contextual bandits with latent confounders" (connections to hidden knowledge state)

**Spaced repetition theory:**
- Pimsleur 1967 — "A Memory Schedule" (graduated interval recall)
- Reddy et al. 2016 — "Unbounded Human Learning: Optimal Scheduling for Spaced Repetition" (KDD)

**Educational data mining:**
- Settles et al. 2018 — "Second Language Acquisition Modeling (SLAM)" shared task (Duolingo dataset)
- Feng et al. 2009 — "ASSISTments dataset" (dataset source)
- Baker & Inventado 2014 — "Educational Data Mining and Learning Analytics" (survey)

**Bayesian experimental design (for BKT-Bandit theory):**
- Chaloner & Verdinelli 1995 — "Bayesian Experimental Design: A Review"
- Lindley 1956 — "On a Measure of Information Provided by an Experiment" (information gain)

**Regret lower bounds:**
- Lai & Robbins 1985 — "Asymptotically efficient adaptive allocation rules" (foundational lower bound)

**Real data baselines:**
- Choffin et al. 2019 — "DAS3H: Modeling Student Learning and Forgetting" (KDD, combines skills + forgetting)
- Wilson et al. 2016 — "Back to the basics: Bayesian extensions of IRT" (Bayesian student modeling)
