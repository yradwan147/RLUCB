# Plan: Parametric-Decay Restless Bandits for Adaptive Learning

## Context

Current results are too weak — marginal improvements, no algorithmic novelty. Just applying Whittle index to education isn't enough (that's EduQate at AAMAS 2025). Accepted NeurIPS papers in this space contribute **new algorithms with provable guarantees under weaker assumptions** or **new problem formulations**.

**The gap we fill**: Existing RMAB learning algorithms (Ortner et al., Wang et al., NeurIPS 2022-2023) treat arm transition kernels as **general unknown matrices** requiring O(S²) parameters per arm. But in education, forgetting dynamics follow **known parametric forms** (exponential, power-law) with only 2-3 parameters (λ, base, α). Nobody has exploited this structure.

**Our novel contribution**: A **Parametric-Decay Restless Bandit (PD-RMAB)** algorithm that:
1. Exploits known parametric structure of forgetting to learn Whittle indices with O(d√T) regret instead of O(S²√T) — exponential improvement
2. Derives **closed-form Whittle index** for the exponential-decay learning model
3. Proves **indexability** of the education-forgetting MDP
4. Shows the derived optimal policy matches and extends cognitive science spacing predictions

This is genuinely novel: no prior work connects parametric forgetting models to structured RMAB learning.

---

## The Novel Algorithm: PD-RMAB

### Problem formulation
- K arms (categories), each with continuous state k_i ∈ [0,1] (knowledge)
- Active transition (quiz): k' = k + α(1-k) with prob p(correct|k) = k
- Passive transition (decay): k' = base + (k - base) · exp(-λ)
- Parameters θ = (α, λ, base) are **unknown but structured** — same parametric family across arms, potentially different parameter values
- Goal: maximize cumulative knowledge ∑_t ∑_i k_i(t), selecting one arm per step

### Key insight
Standard RMAB learning: estimate full S×S transition matrix per arm → O(S² · K) parameters
Our approach: estimate θ = (α, λ, base) per arm → O(d · K) parameters where d=3 << S²

### Algorithm: PD-Whittle
```
1. Initialize: prior over θ_i for each arm i
2. For each round t:
   a. Compute posterior mean θ̂_i from observations
   b. Compute Whittle index W_i(k_i; θ̂_i) using closed-form (see below)
   c. Add exploration bonus: W̃_i = W_i + β_t · σ_i(k_i)
      where σ_i is posterior uncertainty scaled by index sensitivity
   d. Select arm with highest W̃_i
   e. Observe outcome, update posterior on θ_i
```

### Closed-form Whittle index (to derive)
For exponential decay model with parameters (α, λ, base):
- W(k) represents the urgency of quizzing when knowledge is at level k
- Intuition: W(k) should be high when k is low (weakness) AND when k is decaying fast (urgency)
- Expected form: W(k; α, λ, base) = f(1-k, λ, α) where f captures both exploitation and forgetting urgency

### Exploration bonus
- Not standard UCB bonus (arm-count based)
- **Decay-aware**: bonus scales with how fast the arm decays AND how uncertain we are about decay rate
- bonus_i(k, t) = c · √(ln(t)/n_i) · (1 + λ̂_i · t_since_i)
- The λ̂_i · t_since_i term makes exploration more urgent for fast-decaying arms not recently observed

---

## Theoretical Contributions

### Theorem 1: Indexability
Prove that the education-forgetting MDP (asymptotic learning + exponential decay) satisfies Whittle's indexability condition. Use structure: learning is monotone increasing in k, decay is monotone decreasing, and the problem has a threshold structure.

### Theorem 2: Closed-form Whittle index
Derive W(k; α, λ, base) in closed form for the continuous-state model. Show it reduces to:
- Standard UCB-like score when λ → 0 (no forgetting)
- Pure urgency-based scheduling when α → 0 (no learning)

### Theorem 3: Regret bound
PD-Whittle achieves regret R(T) = O(d · K · √T · log T) where d is parameter dimension (d=3 for our model), vs O(S² · K · √T) for general RMAB learning. This is an **exponential improvement** in the state space S.

### Theorem 4: Connection to spacing theory
Show that the derived Whittle index policy produces spacing intervals that match the **expanding retrieval** pattern from cognitive science: optimal spacing between reviews grows exponentially with mastery level.

---

## Implementation Plan

### Phase 1: Whittle Index Computation (~250 lines)
**File: `experiment/whittle.py` (new)**
- Discretize state space: k ∈ {0.01, 0.02, ..., 1.0} (M=100 bins)
- Compute active/passive transition matrices from model params
- Implement adaptive-greedy algorithm for index computation (O(M³))
- Cache index lookup table: W[k_bin] → index value
- Verify: W increases with (1-k) and with λ

### Phase 2: PD-Whittle Selector (~200 lines)
**File: `experiment/selectors.py` (extend)**
- `WhittleIndexSelector(BaseSelector)` — uses precomputed index table
- `PDWhittleSelector(BaseSelector)` — online learning version with:
  - Bayesian posterior on θ_i per arm (conjugate priors for α, λ)
  - Posterior-based exploration bonus
  - Whittle index recomputation as θ estimates improve
- `LinUCBSelector(BaseSelector)` — contextual bandit baseline

### Phase 3: Experiments (~30 jobs)
- Add WhittleIndex, PDWhittle, LinUCB to algorithm registry
- Rerun synthetic sweep: same grid (K, λ, seeds) + new algorithms
- Rerun real data with new algorithms
- **Key experiment**: Vary θ estimation quality to show PD-Whittle's advantage in learning regime

### Phase 4: Theory (pen and paper → LaTeX)
- Indexability proof
- Closed-form index derivation
- Regret bound proof
- Spacing theory connection

### Phase 5: Paper rewrite
- New title: "Parametric-Decay Restless Bandits for Adaptive Learning with Forgetting"
- RMAB formulation + PD-Whittle algorithm + theory + experiments

---

## Why This Is NeurIPS-Level

Comparing to accepted papers:

| Paper | Venue | Contribution type | Our analog |
|-------|-------|-------------------|------------|
| Neural-Q-Whittle | NeurIPS 2023 | Finite-time convergence for neural Whittle | We: structured convergence for parametric decay |
| Index-aware RL | NeurIPS 2022 | Polynomial (not exponential) regret via index structure | We: further reduction via parametric structure |
| Follow-Virtual-Advice | NeurIPS 2023 | Remove UGAP assumption | We: new structural assumption (parametric decay) |
| EduQate | AAMAS 2025 | Network structure in RMAB for education | We: parametric decay structure + learning |

**Our unique position**: We're not just applying RMAB to education (that's EduQate). We're exploiting the **known parametric structure of forgetting** to derive a fundamentally more sample-efficient learning algorithm. This is a new algorithmic contribution, not an application.

---

## Verification

1. **Whittle index sanity**: W(k=0.1) > W(k=0.9), W increases with λ
2. **Closed-form matches numerical**: derived formula matches adaptive-greedy computation
3. **PD-Whittle vs Whittle-known**: PD-Whittle approaches Whittle-known performance as T grows
4. **PD-Whittle vs F-UCB**: PD-Whittle should strictly dominate (it's computing what F-UCB approximates)
5. **Regret curves**: empirical regret falls within theoretical bound
6. **Spacing pattern**: plot inter-review intervals, verify expanding retrieval pattern
7. **Real data**: improvement on both Duolingo and ASSISTments

---

## Critical Files

| File | Changes |
|------|---------|
| `experiment/whittle.py` | NEW — Whittle index computation, MDP discretization |
| `experiment/selectors.py` | Add WhittleIndexSelector, PDWhittleSelector, LinUCBSelector |
| `experiment/selectors.py` | Update SELECTOR_REGISTRY |
| `slurm/submit_new_algos.sh` | NEW — submit experiments with new algorithms |
| `paper/` | Major rewrite (Phase 5, last) |
