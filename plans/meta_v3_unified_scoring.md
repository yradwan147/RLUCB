# Plan: Unified Scoring MetaSelector (No Expert Tracking)

## Context

After 6 iterations of expert-tracking MetaSelector, the core problem is clear: **we can't observe the true reward (knowledge gain) from binary feedback alone**, so any expert-tracking approach biases toward whichever expert has higher correctness (F-UCB) rather than higher knowledge gain (BKT-Bandit).

The solution: **stop trying to select between experts. Instead, fuse the best scoring elements from each algorithm into a single category-level scoring function.** Each base expert still runs in parallel and proposes candidates, but the MetaSelector scores categories directly using a unified formula that combines:

- **BKT-Bandit's strength**: Beta posterior with forgetting → accurate weakness estimation + uncertainty
- **F-UCB's strength**: Explicit forgetting urgency → handles high-decay regime

## The Algorithm: MetaSelector with Unified Category Scoring

### Scoring formula
```
score(i) = (1 - mean_i) + c * std_i * (1 + λ * t_i)
```

Where:
- `mean_i = α_i / (α_i + β_i)` — Beta posterior mean (knowledge estimate)
- `std_i = sqrt(α_i * β_i / ((α_i + β_i)² * (α_i + β_i + 1)))` — posterior uncertainty
- `t_i` — time since last exposure for category i
- `λ` — decay rate from config
- `c` — exploration parameter (1.414)

**Posterior forgetting** (same as BKT-Bandit):
```
decay = exp(-λ * t_i)
α_i ← 1 + (α_i - 1) * decay
β_i ← 1 + (β_i - 1) * decay
```

### Why this should work across all regimes

**At low decay (λ small):**
- `(1 + λ * t_i) ≈ 1` — urgency term is negligible
- Score ≈ `(1 - mean_i) + c * std_i` — **exactly BKT-Bandit**
- Should match BKT-Bandit performance

**At high decay (λ large):**
- `(1 + λ * t_i)` grows fast for stale categories
- Urgency multiplier dominates → forces frequent revisiting
- Score becomes urgency-driven — **similar to F-UCB**
- Should match F-UCB performance

**The key**: the `(1 + λ * t_i)` multiplier on the exploration bonus naturally interpolates between BKT-Bandit behavior (low decay) and F-UCB behavior (high decay) **without needing to detect the regime or track expert performance**.

### Role of base experts
Base experts still run and propose candidate categories. The MetaSelector only considers categories recommended by at least one expert. This filters the action space from K categories to ≤M candidates, improving efficiency. But the final selection is by the unified score, not by expert identity.

If all experts recommend different categories (common early), the unified scorer picks the best one. If experts agree (common late), it follows consensus.

## Implementation

### Changes to MetaSelector in `experiment/selectors.py` (~40 lines changed)

**Replace select_category():**
```python
def select_category(self):
    recs = [b.select_category() for b in self.bases]

    if self.total_questions < self.num_categories:
        # Explore each category once first
        for i in range(self.num_categories):
            if self._cat_attempts[i] == 0:
                return i

    # Apply posterior forgetting
    for i in range(self.num_categories):
        if self._time_since[i] > 0:
            d = math.exp(-self._decay_rate * self._time_since[i])
            self._alpha[i] = 1.0 + (self._alpha[i] - 1.0) * d
            self._beta[i] = 1.0 + (self._beta[i] - 1.0) * d

    # Score all expert-recommended categories with unified formula
    candidates = list(set(recs))
    best_score, best_cat = -1e9, candidates[0]

    for cat in candidates:
        a, b = self._alpha[cat], self._beta[cat]
        mean_k = a / (a + b)
        std = math.sqrt(a * b / ((a+b)**2 * (a+b+1)))
        t = self._time_since[cat]

        weakness = 1 - mean_k
        urgency_multiplier = 1.0 + self._decay_rate * t
        score = weakness + self._exploration * std * urgency_multiplier

        if score > best_score:
            best_score, best_cat = score, cat

    return best_cat
```

**Replace update():**
```python
def update(self, category, correct):
    for b in self.bases:
        b.update(category, correct)
    if correct:
        self._alpha[category] += 1
    else:
        self._beta[category] += 1
    self._cat_attempts[category] += 1
    self.total_questions += 1
    self._time_since += 1
    self._time_since[category] = 0
```

### Init changes
Replace all the expert-tracking state (`_expert_counts`, `_expert_reward`, `_expert_correct`, `_tenure_*`, `_category_knowledge`, `_ewma_alpha`) with simple:
```python
self._alpha = np.ones(num_categories)  # Beta posterior
self._beta = np.ones(num_categories)
self._time_since = np.zeros(num_categories, dtype=np.int32)
self._cat_attempts = np.zeros(num_categories, dtype=np.int32)
self._decay_rate = 0.01  # set from config
self._exploration = 1.414  # set from config
```

### Registry change
Pass decay_rate and exploration_param from config:
```python
"meta": lambda cfg, rng: MetaSelector(
    base_selectors=[...],
    num_categories=cfg.num_categories,
    decay_rate=cfg.decay_rate,
    exploration_param=cfg.exploration_param,
    rng=rng,
)
```

### Also create slurm script
`slurm/submit_meta_v3.sh` — 42 jobs (36 synthetic + 6 real data)

## Critical File
- `experiment/selectors.py` — MetaSelector class (lines ~1205-1373)

## Verification
1. **K=6 d=0.005**: should match BKT-Bandit (score ≈ BKT formula when λ*t small)
2. **K=6 d=0.05**: should match F-UCB (urgency term dominates)
3. **K=20 d=0.005**: no failure mode (no expert tracking to go wrong)
4. **All configs**: top-3 consistently, gap <5% to best
5. **Quick test**: 30 students, 2000 questions, 6 configs locally
6. **Then IBEX**: 42 jobs for full validation

## Theoretical Soundness (Audit vs Orabona's Notes)

**Connection to UCB theory**: The formula `(1-mean) + c * std * (1+λt)` is a valid Upper Confidence Bound where:
- `(1-mean)` is the empirical weakness estimate (exploitation term)
- `c * std` is the posterior confidence width (standard UCB exploration)
- `(1+λt)` is a time-varying confidence multiplier for non-stationary arms

**Connection to Optimistic OMD (Orabona §6.10)**: The urgency multiplier acts as a "hint" about future losses — categories not quizzed recently will have higher future loss (due to forgetting). This is analogous to Optimistic OMD where g̃_{t+1} predicts the next gradient.

**Connection to posterior forgetting**: The Beta posterior shrinkage `α ← 1 + (α-1)*decay` effectively increases the confidence width over time, which is the theoretically correct response to non-stationarity. The explicit `(1+λt)` multiplier reinforces this — it's a belt-and-suspenders approach.

**Why this is NOT just a heuristic**: Unlike the expert-tracking MetaSelector which had no clear loss function, this formula has a well-defined interpretation: maximize the UCB index over categories, where the confidence width accounts for forgetting-induced non-stationarity. The regret guarantee follows from standard UCB analysis with time-varying confidence bounds (see Garivier & Moulines 2011 for non-stationary UCB).

**What the experts provide**: The base experts filter the K categories to ≤M candidates. From LEA theory (Orabona §6.8), using expert advice to restrict the action set reduces regret from O(√(TK)) to O(√(TM)) — a concrete improvement.

## Why This Is Different From Previous Attempts
- **No expert tracking** — eliminates the reward signal problem entirely
- **No reward estimation** — no EWMA, no ΔK, no correctness-based UCB
- **Deterministic** — no Thompson-style noise, no randomized selection
- **Mathematically interpolates** between BKT-Bandit (λ→0) and urgency-driven (λ→∞)
- **Uses experts only for candidate filtering** — not for credit assignment
