# Plan: Fix MetaSelector Reward Signal — Knowledge Gain instead of Correctness

## Context

MetaSelector uses correctness rate to evaluate experts via UCB. But the best expert (BKT-Bandit) targets weak categories, getting LOWER accuracy but producing HIGHER knowledge gain. Result: MetaSelector biases toward F-UCB (high accuracy) and never beats BKT-Bandit in its regimes. Root cause confirmed: Oracle has lowest accuracy (0.527) but highest knowledge (0.581).

Research identified the solution: use **knowledge state estimation** from binary responses, then reward experts based on **ΔK (knowledge improvement)** rather than correctness level.

## Approach: EWMA Knowledge Estimation + ΔK Reward

Maintain per-category knowledge estimates via EWMA of correctness with forgetting decay. Reward each expert based on the change in mean estimated knowledge when that expert was followed.

### Why this works
- Expert targets weak category → student gets it wrong → K_hat stays low BUT...
- Next time student answers that category correctly → K_hat jumps up → large ΔK
- Expert that avoids weak categories → K_hat flat → near-zero ΔK
- Expert that prevents decay → K_hat doesn't drop → positive ΔK from decay avoidance

### Changes to MetaSelector (in `experiment/selectors.py`)

**Add to `__init__`:**
```python
self._category_knowledge = np.full(num_categories, 0.5)  # EWMA knowledge estimate
self._ewma_alpha = 0.2  # smoothing factor
```

**Replace reward computation in `update()`:**
```python
# Compute knowledge BEFORE update
k_before = float(np.mean(self._category_knowledge))

# Update EWMA for quizzed category
self._category_knowledge[category] = (
    self._ewma_alpha * (1.0 if correct else 0.0)
    + (1 - self._ewma_alpha) * self._category_knowledge[category]
)

# Apply forgetting decay to non-quizzed categories
decay = 0.999  # slight decay per step
for i in range(self.num_categories):
    if i != category:
        self._category_knowledge[i] *= decay

# Compute knowledge AFTER update
k_after = float(np.mean(self._category_knowledge))

# Reward = knowledge gain (can be negative if decay > learning)
delta_k = k_after - k_before

# UCB tracking: accumulate ΔK instead of correctness
self._expert_reward[chosen] += delta_k
self._expert_counts[chosen] += 1
```

**Replace UCB scoring in `select_category()`:**
```python
scores[i] = self._expert_reward[i] / n  # mean ΔK, not correctness rate
scores[i] += math.sqrt(2 * math.log(self.total_questions + 1) / n)
```

**Add to `reset()`:**
```python
self._category_knowledge = np.full(self.num_categories, 0.5)
self._expert_reward = np.zeros(self.M)
```

### Also add slurm submission script
`slurm/submit_meta_v2.sh` — 42 jobs (36 synthetic + 6 real data)

## Critical File
- `experiment/selectors.py` — MetaSelector class (lines ~1214-1325)

## Verification
1. At K=6 d=0.005: meta should follow BKT-Bandit (not F-UCB) → rank #1-2
2. At K=6 d=0.05: meta should follow F-UCB → rank #1-2
3. K=20 d=0.005: should fix the rank-12 failure mode
4. Gap to best should be <3% across all configs
5. Quick local test: 30 students, 2000 questions, 6 configs
