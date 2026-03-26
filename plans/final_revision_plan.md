# Plan: Final Paper Revision — Theory + Clarity + Narrative Polish

## Context

Three reviews (NeurIPS: borderline reject, ICML: revise, ICLR: lean accept). All praise the same strengths: RMAB framing, Greedy-Min catastrophe, broad experiments. The #1 universal weakness is **no theory**. Secondary issues are MetaSelector confusion, Whittle terminology, and caption inconsistencies. User wants: (1) add regret bounds, (2) keep confident tone, (3) move evolution details to supplementary, (4) incorporate all useful insights from worklog/memory.

**Target narrative after revision:**
1. *Problem*: Forgetting breaks standard bandits → RMAB is the right framework
2. *Theory*: We PROVE myopic policies fail (Theorem 1), and our algorithms converge (Theorems 2-3)
3. *Algorithms*: F-UCB, BKT-Bandit, Advantage Index, MetaSelector — clean, one-formula-each descriptions
4. *Experiments*: Comprehensive validation confirms theory + reveals additional insights
5. *Contribution*: Both theoretical foundations AND practical algorithms for forgetting-aware selection

This transforms the paper from "benchmark paper" to "principled algorithmic contribution with theory."

---

## Part 1: Theory (NEW — addresses #1 weakness from all 3 reviews)

### Theorem 1: Greedy-Min Catastrophe Lower Bound
**Statement**: Any myopic policy π_greedy = argmin_i k_i suffers regret Ω(Kλ T) relative to balanced allocation.
**Proof approach**: Dynamical-systems argument. K identical arms, balanced quizzing loses O(λT) to decay, greedy-min loses Ω(KλT) because K-1 arms decay each step while only 1 is quizzed.
**Numerical verification**: Python — run K arms at k=0.5, compare greedy-min vs round-robin total knowledge. Gap should grow linearly with K and λ.
**Paper location**: New §3.5 "Theoretical Analysis" (proof sketch in main, full proof in supplementary)

### Theorem 2: BKT-Bandit Bayesian Regret Bound
**Statement**: BKT-Bandit achieves Bayesian regret O(K√(T log T)) under forgetting dynamics with effective sample size n_eff = Σ exp(-λ(t-s)).
**Proof approach**: Beta posterior concentration under geometric discounting. Standard Bayesian regret decomposition (Russo & Van Roy 2014 style) with n_eff replacing n.
**Numerical verification**: Track |posterior_mean - true_k| over time. Should decay as 1/√n_eff.
**Paper location**: §3.5, proof in supplementary

### Theorem 3: F-UCB Regret Bound
**Statement**: F-UCB with known λ achieves regret O(K√T) relative to best fixed-category oracle.
**Proof approach**: Phase-dependent UCB analysis. The exp(-λt) decay ensures stale estimates don't persist — bounded number of "effective regime changes" per arm. Adapt Auer et al. 2002 machinery.
**Numerical verification**: Plot cumulative regret vs √T. Should be sublinear.
**Paper location**: §3.5, proof in supplementary

### Theory section connection to Orabona
Cite Orabona's framework (§6.10 Optimistic OMD): MetaSelector's unified scoring is a UCB with non-stationary confidence width, which connects to Optimistic OMD where the urgency term acts as a "hint" about future losses. Expert candidate filtering reduces the action set from K to ≤M (§6.8 LEA reduction).

---

## Part 2: MetaSelector Cleanup (addresses #1 reviewer complaint)

**Problem**: Lines 132-149 of sec/3_method.tex describe TWO algorithms: first UCB-over-experts (Eq 9), then unified scoring (Eq 10). All 3 reviewers flag this as confusing.

**Fix**:
- **Main paper**: Describe ONLY the unified scoring approach (Eq 10). MetaSelector runs M base selectors to propose candidate categories, then scores each candidate with:
  ```
  s_i = (1 - μ_i)·exp(-λt_i) + 0.5·(1 - exp(-λt_i)) + c·σ_i
  ```
  This combines F-UCB's three-term structure with BKT-Bandit's posterior. No expert tracking.
- **Remove**: Eq 9 (UCB-over-experts formula), lines about "mean rewards" and "counts for each expert"
- **Supplementary**: Add "MetaSelector Design Evolution" section explaining the journey from expert-tracking (v1-v2) to unified scoring (v3), and why expert tracking failed (reward signal mismatch: correctness is inversely correlated with knowledge gain)

---

## Part 3: Whittle → Advantage Index Rename

**Fix**: Replace "Whittle-advantage" / "Whittle" in algorithm names with "Advantage Index" (or "Active-Passive Advantage") throughout. Keep "Whittle" only when citing Whittle 1988 or discussing the RMAB framework.

**Files**: sec/3_method.tex (algorithm description), tables/table_consistency.tex, sec/supplementary.tex, abstract, intro

---

## Part 4: Caption + Figure Fixes

### 4a. fig_scaling caption
Current: "Lookahead Oracle Recovers at K≥20"
Fix: "Scaling at λ=0.01: all methods degrade as K grows. Lookahead Oracle (brown dashed) outperforms Greedy-Min at K≥20 under moderate-to-high decay."

### 4b. Typo grep
Grep all tex files for "Traiertorios", "K-61-001", "Decay-UCB" and fix.

### 4c. Check all figure titles in the actual PNGs
If any matplotlib titles contain typos, regenerate those figures.

---

## Part 5: Statistical Significance

Add ± std (from 10-seed runs) to main results table for K=6. Add sentence: "At d=0.01, the top-4 algorithms (Greedy-Min, PD-Whittle, MetaSelector, BKT-Bandit) have non-overlapping 95% CIs versus Random, confirming statistical significance."

---

## Part 6: Minor Fixes

- Objective clarification: "We report final mean knowledge as the primary metric; AUC (time-averaged knowledge) in the supplementary shows consistent rankings."
- Remove any self-deprecating language ("preliminary," "tentative," "we emphasize limitations")
- Ensure "Decay-UCB" doesn't appear anywhere (should be F-UCB)
- Clean bib entries (entry types already fixed, just verify)

---

## Implementation Order

1. **Derive and verify theory** (Theorems 1-3) — numerical checks first, then LaTeX
2. **Rewrite MetaSelector** in sec/3_method.tex — one formula, one description
3. **Rename Whittle → Advantage Index** — global find/replace
4. **Add Theory section** (§3.5) with theorem statements + proof sketches
5. **Fix captions, typos, figures**
6. **Add ± std to tables**
7. **Move MetaSelector evolution to supplementary**
8. **Full proofs in supplementary**
9. **Recompile, verify, commit**

---

## Files to Modify

| File | Changes |
|------|---------|
| `sec/3_method.tex` | MetaSelector rewrite, Whittle→Advantage, NEW §3.5 Theory |
| `sec/0_abstract.tex` | Add "prove" language, Whittle→Advantage |
| `sec/1_intro.tex` | Update contributions (add theory), Whittle→Advantage |
| `sec/4_experiments.tex` | Fix captions, add significance, objective note |
| `sec/supplementary.tex` | MetaSelector evolution, full proofs for 3 theorems |
| `tables/table_main_results.tex` | Add ± std |
| `tables/table_consistency.tex` | Whittle→Advantage |
| `refs.bib` | Verify clean |
| `paper/figs/` | Regenerate if matplotlib titles have typos |

## Verification

1. Three theorems stated with proof sketches in §3.5
2. Numerical verification matches theoretical predictions
3. Full proofs in supplementary
4. MetaSelector: ONE formula (Eq 10 only), clear description
5. No "Whittle" in algorithm names
6. All captions match figure content
7. ± std in main table
8. No typos/artifacts
9. No self-deprecating language
10. Paper compiles cleanly ≤20 pages
