# Plan: Meta-Selector via Online Expert Aggregation

## Context

No single algorithm dominates all regimes: BKT-Bandit wins at K≥20 d≤0.01, F-UCB wins at d=0.05, Whittle wins at K=6 low decay. A meta-algorithm that adaptively combines the best selectors would achieve near-best-expert performance across ALL regimes — solving the regime-split problem.

The user studied online learning and suggests expert aggregation (Hedge/EXP). Research confirms **CORRAL** (Agarwal et al. 2017, "Corralling a Band of Bandit Algorithms") is the ideal framework — specifically designed for combining bandit algorithms with log(M) regret overhead.

## The Algorithm: MetaSelector

### Core idea
Run M base selectors in parallel. Each step, a master algorithm picks which base selector's recommendation to follow. Track performance and shift weight toward the best-performing selector. Guarantee: cumulative performance within O(√T log M) of the best base selector in hindsight.

### Implementation approach: Exponential Weights (EXP4-style)

We use EXP4 rather than full CORRAL for simplicity and because our selectors provide deterministic recommendations (not probability distributions):

```
1. Initialize: uniform weights w_i = 1/M for M base selectors
2. Each step t:
   a. Each base selector i recommends category a_i
   b. Form mixture: p(category j) = Σ_i w_i · 1[a_i == j]
   c. Sample category from p (or pick argmax for exploitation)
   d. Observe outcome (correct/incorrect)
   e. Update all base selectors with (category, outcome)
   f. Compute reward for each selector based on outcome:
      - If selector i recommended the CHOSEN category: use observed reward
      - If selector i recommended a DIFFERENT category: use importance-weighted estimate
   g. Update weights: w_i *= exp(η · estimated_reward_i)
   h. Normalize weights
```

### Key design choices

1. **Which base selectors to include**: The top performers from each regime:
   - BKT-Bandit (best at K≥20 d≤0.01)
   - F-UCB (best at d=0.05)
   - PD-Whittle (best equity, good at low decay)
   - Leitner (consistently top-5)
   - Random (surprisingly competitive baseline)

2. **Reward signal**: Use knowledge-based reward rather than correctness:
   - Reward = 1 if selector chose a category where student IMPROVED
   - Or simpler: reward = 1 - estimated_knowledge[chosen_category] (weakness targeting)

3. **Learning rate**: η = √(ln(M) / T) for optimal regret, or adaptive

4. **Warm-up period**: Run all selectors equally for first ~K steps to collect initial data

### Theoretical guarantee
MetaSelector achieves regret R(T) ≤ R_best(T) + O(√(T · ln M))
where R_best is the regret of the best base selector. With M=5 base selectors, the overhead is √(T · ln 5) ≈ 1.27√T — negligible for large T.

---

## Implementation

### File: `experiment/selectors.py` — add MetaSelector class (~100 lines)

```python
class MetaSelector(BaseSelector):
    """Online expert aggregation over base selectors.
    Uses exponential weights to track and combine M base algorithms."""

    def __init__(self, base_selectors: List[BaseSelector],
                 learning_rate: float = 0.1, ...):
        self.bases = base_selectors
        self.weights = np.ones(len(base_selectors)) / len(base_selectors)
        self.eta = learning_rate
        ...

    def select_category(self):
        # Get recommendations from all base selectors
        recommendations = [b.select_category() for b in self.bases]
        # Weight-vote: pick category with highest total weight
        votes = np.zeros(self.num_categories)
        for i, cat in enumerate(recommendations):
            votes[cat] += self.weights[i]
        return int(np.argmax(votes))

    def update(self, category, correct):
        # Update all base selectors
        for b in self.bases:
            b.update(category, correct)
        # Compute reward for each based on their recommendation
        reward = 1.0 if correct else 0.0
        for i, b in enumerate(self.bases):
            rec = self._last_recommendations[i]
            if rec == category:
                self.weights[i] *= np.exp(self.eta * reward)
            # Don't penalize selectors that recommended different categories
        # Normalize
        self.weights /= self.weights.sum()
```

### Registry entry
```python
"meta": lambda cfg, rng: MetaSelector(
    base_selectors=[
        create_selector("bkt_bandit", cfg, rng),
        create_selector("fucb", cfg, rng),
        create_selector("pd_whittle", cfg, rng),
        create_selector("leitner", cfg, rng),
        create_selector("random", cfg, rng),
    ],
    num_categories=cfg.num_categories,
)
```

### Simulation integration
- MetaSelector implements BaseSelector interface — drops in with no simulation changes
- Each base selector tracks its own state independently
- Meta layer only decides which recommendation to follow

---

## Verification

1. **Sanity check**: MetaSelector should never be worse than random (the weakest base)
2. **Convergence**: After enough steps, weights should concentrate on the best base selector for the current (K, decay) regime
3. **Across regimes**: MetaSelector should be top-3 in ALL configs (not just some)
4. **Weight inspection**: Log which base selector has highest weight at end of each run — verify it matches the known winner for that regime
5. **Compare to best**: Gap to best single algorithm should be small (< 5%)

---

## IBEX Submission

### `slurm/submit_meta_experiments.sh`
Runs MetaSelector against all existing algorithms on both synthetic and real data.

**Synthetic**: 4 K × 3 decay × 3 seeds = 36 jobs (each runs all 14 algorithms)
**Real data**: 2 datasets × 3 seeds = 6 jobs

Total: 42 jobs. Same slurm runners (`run_experiment.sh`, `run_real_data.sh`) — just picks up the new `meta` algorithm from the registry via `--all-algorithms`.

```bash
#!/bin/bash
set -e
mkdir -p slurm/slurm_logs results

SEEDS="42 123 456"
COUNT=0

# Synthetic sweep (36 jobs)
for K in 6 20 50 100; do
    for DECAY in 0.005 0.01 0.05; do
        for SEED in $SEEDS; do
            sbatch -J meta_k${K}_d${DECAY}_s${SEED} \
                slurm/run_experiment.sh $K $DECAY $SEED
            COUNT=$((COUNT + 1))
        done
    done
done

# Real data (6 jobs)
for DS in duolingo assistments; do
    for SEED in $SEEDS; do
        sbatch -J meta_rd_${DS}_s${SEED} \
            slurm/run_real_data.sh $DS $SEED both
        COUNT=$((COUNT + 1))
    done
done

echo "Submitted $COUNT jobs (14 algorithms including MetaSelector)"
```

---

## Critical Files

| File | Changes |
|------|---------|
| `experiment/selectors.py` | Add MetaSelector class + registry entry |
| `slurm/submit_meta_experiments.sh` | NEW — submit 42 jobs (synthetic + real) |
| No other files need changes | Drops in via BaseSelector interface |

---

## Expected Impact

If MetaSelector works as theory predicts:
- It should be **top-3 in all 12 configs** (no other algorithm achieves this)
- At low decay: inherits PD-Whittle/BKT-Bandit performance
- At high decay: inherits F-UCB performance
- Overhead is ~√(T ln 5) which is negligible for T=10,000

This is a strong paper contribution: **a principled meta-algorithm with provable guarantees that adapts to any forgetting regime**.
