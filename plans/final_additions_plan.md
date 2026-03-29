# Plan: Four Additions to Strengthen the Paper

## Context

Paper is at 8.1/10. Targeting NeurIPS 2026 (May 6 deadline, ~5 weeks). Four additions identified from comparison with top venue papers, ordered by impact.

---

## Item 1: Matching Lower Bound + Tightened Upper Bound

### Goal
Prove a lower bound for the forgetting bandit and reconcile with the upper bound. The current upper bound O(K√T) may be loose — the true rate is likely Θ(√(KT)), which is standard for K-armed bandits.

### Approach

**Step 1: Prove Ω(√(KT)) lower bound via reduction (1 paragraph)**
- Any forgetting-bandit instance with λ→0 reduces to a standard K-armed stochastic bandit
- The minimax lower bound for K-armed bandits is Ω(√(KT)) (Lattimore & Szepesvári 2020, Theorem 15.2)
- Therefore, the forgetting bandit inherits Ω(√(KT))

**Step 2: Prove forgetting-specific additive penalty Ω(λT) (1 paragraph)**
- During exploration, the optimal arm decays while idle
- If the learner explores K arms equally, arm i* is idle for (K-1)/K of the time
- Each idle step costs λ·(k* - k_base) in regret vs the fixed-arm comparator
- Total forgetting penalty: Ω(λ·T) when λ > √(K/T)

**Step 3: Combined lower bound**
```
R_T(π) ≥ Ω(max(√(KT), λ·T))
```
- Low λ regime: Ω(√(KT)) dominates (standard bandit hardness)
- High λ regime: Ω(λT) dominates (forgetting penalty)

**Step 4: Tighten upper bound for F-UCB**
The current proof of O(K√T) has a loose step in the regime-counting argument. Tighten to:
```
R_T(F-UCB) ≤ O(√(KT·log T))
```
This matches the lower bound up to log factors.

**Step 5: Numerical verification**
- Run F-UCB on K∈{6,20,50,100}, T∈{1000,5000,10000,50000}, λ∈{0.005,0.01,0.05}
- Compute regret vs best fixed arm
- Plot regret/√(KT) — should be roughly constant if Θ(√(KT)) is correct
- Plot regret/(K√T) — should decrease with K if O(K√T) is too loose

### Paper changes
- **Theorem 3**: Update statement from O(K√T) to O(√(KT log T))
- **New Theorem 4**: Lower bound Ω(max(√(KT), λT))
- **Proof sketch**: 2-3 sentences in main text
- **Full proof**: In supplementary (extend existing proof section)
- **Figure**: regret scaling plot in supplementary showing the bound is tight

### Files to modify
- `paper/sec/3_method.tex` — update Theorem 3 statement, add Theorem 4
- `paper/sec/supplementary.tex` — full proofs, numerical verification figure

### Verification
- Numerical simulation matches theoretical prediction
- Upper and lower bounds match up to log factors
- No contradiction with existing experimental results

---

## Item 2: Equity-Aware Algorithm (EquiBandit)

### Goal
A new algorithm that explicitly balances knowledge equity (low cross-category variance) with average knowledge. Addresses the finding that BKT-Bandit has Gini=0.449 at K=20.

### Algorithm Design

**EquiBandit** = BKT-Bandit + variance penalty:
```python
score_i = (1 - μ_i) + c·σ_i + η·(μ_i - μ̄)²
```
Where:
- `(1 - μ_i)` = weakness (same as BKT-Bandit)
- `c·σ_i` = uncertainty exploration (same as BKT-Bandit)
- `η·(μ_i - μ̄)²` = equity bonus: categories far from the mean get priority
- `μ̄ = mean(μ_1, ..., μ_K)` = average posterior mean across all categories
- `η` = equity weight (hyperparameter, default 0.5)

This is equivalent to adding a regularizer that pulls all categories toward equal mastery.

### Implementation
- New class `EquiBanditSelector(BaseSelector)` in `experiment/selectors.py` (~60 lines)
- Uses same Beta posterior with forgetting as BKT-Bandit
- Additional: tracks mean knowledge across categories for the equity term
- Registry entry: `"equi_bandit"` with η=0.5

### Experiments needed
- **Local test**: K=6,20 × λ=0.005,0.01,0.05 × 30 students × 2000 questions (quick)
- **IBEX run**: Same configs as main experiments (add to next IBEX batch)
- **Metrics**: avg_knowledge, weakest_category, Gini coefficient, knowledge_variance
- **Key comparison**: EquiBandit vs BKT-Bandit — does it achieve similar avg_knowledge with much better equity?

### Paper changes
- **New paragraph** in §3.3 (Novel Algorithms) describing EquiBandit
- **Pareto analysis figure**: Plot avg_knowledge vs Gini for all algorithms — EquiBandit should be on the Pareto frontier
- **Table update**: Add EquiBandit row to main results + consistency table
- **Discussion**: equity-efficiency tradeoff is now explicitly addressed by an algorithm, not just observed

### Files to modify
- `experiment/selectors.py` — add EquiBanditSelector + registry entry
- `paper/sec/3_method.tex` — new paragraph
- `paper/sec/4_experiments.tex` — new results/figure
- `paper/tables/table_main_results.tex` — add row
- `paper/tables/table_consistency.tex` — add row

### Verification
- EquiBandit achieves Gini < 0.1 at K=20 (vs BKT-Bandit's 0.449)
- EquiBandit's avg_knowledge within 5% of BKT-Bandit
- Pareto figure shows EquiBandit dominates on the equity-efficiency frontier

---

## Item 3: Comparison Table (Prior Work)

### Goal
A checkmark table in the introduction comparing ForgetBandit vs prior work across key features. Standard NeurIPS pattern.

### Design

| Work | Year | Forgetting Model | RMAB | Theory | # Algos | Real Data | Code |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Clément et al. | 2015 | ✗ | ✗ | ✗ | 2 | ✗ | ✗ |
| Reddy et al. | 2016 | ✓ | ✓ | ✓ | 1 | ✗ | ✗ |
| Tabibian et al. | 2019 | ✓ | ✗ | ✓ | 1 | ✗ | ✗ |
| EduQate (Mate) | 2024 | ✗ | ✓ | ✗ | 3 | ✓ | ✗ |
| **ForgetBandit (Ours)** | 2026 | **✓** | **✓** | **✓** | **17** | **✓** | **✓** |

### Implementation
- New table file: `paper/tables/table_comparison.tex`
- Place after contribution list in intro (or at end of related work)
- Use `\cmark`/`\xmark` formatting from NeurIPS template

### Files to modify
- `paper/tables/table_comparison.tex` — NEW
- `paper/sec/1_intro.tex` — reference the table

---

## Item 4: Decision Flowchart / Regime Recommendation

### Goal
A practical figure showing practitioners which algorithm to use based on their (K, λ) regime. Makes the paper actionable.

### Design

```
                    ┌── λ ≤ 0.01? ──┐
                    │               │
              ┌─ YES ─┐       ┌─ NO (high decay) ─┐
              │       │       │                    │
         K ≤ 20?    K > 20?   Use F-UCB
              │       │       (urgency dominates)
        ┌─YES─┐  ┌─YES─┐
        │     │  │     │
   BKT-Bandit  MetaSelector
   (posterior    (stable rank 3,
    tracking)    no regime knowledge
                 needed)
```

With the catastrophe threshold from Theorem 1 annotated:
**K > 1 + (α-β)k₀(1-k₀)/(λ(k₀-base)) → avoid Greedy-Min!**

### Implementation
- Generate as matplotlib figure: `paper/figs/fig_flowchart.png`
- Or create as TikZ in LaTeX (cleaner for publication)
- Place in experiments section (§4.6 Regime Analysis) or conclusion

### Files to modify
- `paper/figs/fig_flowchart.png` — NEW (generated via matplotlib or TikZ)
- `paper/sec/4_experiments.tex` or `paper/sec/5_conclusion.tex` — reference

---

## Implementation Order

1. **Item 2 (EquiBandit)** — implement code first, test locally, then IBEX
2. **Item 1 (Lower bound)** — numerical verification first, then write proofs
3. **Item 3 (Comparison table)** — quick LaTeX, no experiments
4. **Item 4 (Flowchart)** — quick figure generation
5. **IBEX run** for EquiBandit + lower bound verification
6. **Integrate all into paper** — update tex files
7. **Recompile and final review**

### IBEX Jobs Estimate
- EquiBandit validation: 6 jobs (K=6,20 × 3 decay × seed=42)
- Lower bound verification: 12 jobs (K=6,20,50,100 × 3 λ × seed=42, short runs with varying T)
- **Total: ~18 jobs**

---

---

## Item 5: Algorithm Count Audit — Drop Redundant Variants

### Problem
17 algorithms risks looking like count-padding. Top NeurIPS/ICML papers typically have 4-8 methods. PD-Advantage and Adapt-Advantage perform within 0.004 of base Advantage Index (verified: K=6 d=0.01, all three at 0.579-0.583, completely indistinguishable).

### Action
**Drop PD-Advantage and Adapt-Advantage.** Keep only base Advantage Index. Add one sentence in supplementary: "We tested posterior-based and adaptive estimation variants of the Advantage Index with negligible performance differences; we report the base version."

### Final algorithm count: 14
- 7 baselines: Random, UCB1, Thompson, ε-Greedy, SW-UCB, Leitner, Discounted TS
- 2 references: Greedy-Min, Lookahead Oracle
- 5 novel: F-UCB, BKT-Bandit, Advantage Index, MetaSelector, EquiBandit

This is a credible count where every algorithm has a distinct, justified role.

### Files to modify
- Remove `pd_whittle` and `adaptive_whittle` from tables (table_main_results, table_consistency, table_scaling)
- Update "16 algorithms" → "14 algorithms" in abstract, intro, experiments
- Add note about dropped variants in supplementary

---

## Verification (after all 5 items)

1. Lower bound Ω(√(KT)) proven and matches numerical simulation
2. Upper bound tightened to O(√(KT log T)) — consistent with experiments
3. EquiBandit achieves Gini < 0.1 with <5% avg_knowledge loss vs BKT-Bandit
4. Pareto figure clearly shows equity-efficiency frontier
5. Comparison table accurately represents all prior work
6. Flowchart is clear and matches experimental findings
7. Paper compiles cleanly, page count still ≤10 main
8. Algorithm count is 14 — each with a distinct, justified role
9. No "count-padding" impression — baselines are established methods, novel algs are distinct
