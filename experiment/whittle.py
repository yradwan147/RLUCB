"""
Whittle index computation for the education-forgetting restless bandit.

Each arm (category) has continuous knowledge state k ∈ [0,1]:
  - Active (quiz): k' = k + α(1-k) if correct, k' = k - β·k if incorrect
                   P(correct) = k
  - Passive (no quiz): k' = base + (k - base) · exp(-λ)

We discretize the state space and compute the Whittle index via the
adaptive-greedy algorithm (Niño-Mora 2007).

The Whittle index W(k) represents the subsidy at which an arm in state k
is indifferent between being quizzed and not. Policy: quiz the arm with
highest W(k_i).
"""

import math
from typing import Optional, Tuple

import numpy as np


def build_transition_matrices(
    num_states: int,
    learning_rate: float,
    incorrect_penalty: float,
    decay_rate: float,
    base_knowledge: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build active and passive transition matrices for the discretized MDP.

    Args:
        num_states: Number of discrete knowledge bins (M)
        learning_rate: α — knowledge gain on correct answer
        incorrect_penalty: β — knowledge loss on incorrect answer
        decay_rate: λ — exponential decay rate per timestep
        base_knowledge: minimum knowledge floor

    Returns:
        states: array of knowledge values [k_0, k_1, ..., k_{M-1}]
        P_active: M×M active transition matrix (when quizzed)
        P_passive: M×M passive transition matrix (when not quizzed)
    """
    states = np.linspace(0.01, 0.99, num_states)
    P_active = np.zeros((num_states, num_states))
    P_passive = np.zeros((num_states, num_states))

    for i, k in enumerate(states):
        # Active transition: quiz this category
        # With prob k: correct → k' = k + α(1-k)
        k_correct = k + learning_rate * (1 - k)
        k_correct = max(0.01, min(0.99, k_correct))
        j_correct = _nearest_state(states, k_correct)

        # With prob (1-k): incorrect → k' = k - β·k
        k_incorrect = k - incorrect_penalty * k
        k_incorrect = max(0.01, min(0.99, k_incorrect))
        j_incorrect = _nearest_state(states, k_incorrect)

        P_active[i, j_correct] += k        # prob correct
        P_active[i, j_incorrect] += 1 - k  # prob incorrect

        # Passive transition: don't quiz → forgetting
        k_decay = base_knowledge + (k - base_knowledge) * math.exp(-decay_rate)
        k_decay = max(0.01, min(0.99, k_decay))
        j_decay = _nearest_state(states, k_decay)
        P_passive[i, j_decay] = 1.0  # deterministic decay

    return states, P_active, P_passive


def _nearest_state(states: np.ndarray, value: float) -> int:
    """Find index of nearest state to given value."""
    return int(np.argmin(np.abs(states - value)))


def compute_whittle_indices(
    states: np.ndarray,
    P_active: np.ndarray,
    P_passive: np.ndarray,
    reward_active: Optional[np.ndarray] = None,
    reward_passive: Optional[np.ndarray] = None,
    discount: float = 0.99,
) -> np.ndarray:
    """
    Compute Whittle indices via the adaptive-greedy algorithm.

    For each state s, the Whittle index W(s) is the subsidy β at which
    the arm is indifferent between active and passive.

    Uses value iteration on the single-arm problem with subsidy.

    Args:
        states: array of M state values
        P_active: M×M active transition matrix
        P_passive: M×M passive transition matrix
        reward_active: M-vector of rewards when active (default: knowledge gain)
        reward_passive: M-vector of rewards when passive (default: 0)
        discount: discount factor γ

    Returns:
        W: M-vector of Whittle indices, one per state
    """
    M = len(states)

    if reward_active is None:
        # Reward = knowledge state value (same for both actions)
        # We want to maximize cumulative knowledge. The immediate reward
        # is the current knowledge level regardless of action taken.
        # The Whittle index captures the DIFFERENCE in long-term value
        # between quizzing (active) and not quizzing (passive).
        reward_active = states.copy()

    if reward_passive is None:
        # Same reward for passive — the state itself is the reward.
        # The subsidy β is added to passive reward during index computation.
        reward_passive = states.copy()

    # Binary search for the subsidy that makes each state indifferent
    W = np.zeros(M)

    for s in range(M):
        lo, hi = -10.0, 10.0

        for _ in range(80):  # binary search iterations
            beta = (lo + hi) / 2.0

            # Compute optimal policy for single arm with subsidy beta
            # V(s) = max(R_active(s) + γ P_active V, R_passive(s) + β + γ P_passive V)
            V = np.zeros(M)
            for _ in range(200):  # value iteration
                Q_active = reward_active + discount * P_active @ V
                Q_passive = reward_passive + beta + discount * P_passive @ V
                V_new = np.maximum(Q_active, Q_passive)
                if np.max(np.abs(V_new - V)) < 1e-8:
                    break
                V = V_new

            # At state s, check if passive is preferred
            q_active = reward_active[s] + discount * P_active[s] @ V
            q_passive = reward_passive[s] + beta + discount * P_passive[s] @ V

            if q_passive >= q_active:
                # Subsidy too high → passive preferred → decrease
                hi = beta
            else:
                # Subsidy too low → active preferred → increase
                lo = beta

        W[s] = (lo + hi) / 2.0

    return W


def compute_whittle_index_table(
    learning_rate: float = 0.12,
    incorrect_penalty: float = 0.02,
    decay_rate: float = 0.01,
    base_knowledge: float = 0.10,
    num_states: int = 100,
    discount: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a lookup table mapping knowledge state → Whittle index.

    Args:
        learning_rate, incorrect_penalty, decay_rate, base_knowledge: model params
        num_states: discretization granularity
        discount: discount factor

    Returns:
        states: M-vector of knowledge values
        indices: M-vector of Whittle indices
    """
    states, P_active, P_passive = build_transition_matrices(
        num_states, learning_rate, incorrect_penalty, decay_rate, base_knowledge
    )

    indices = compute_whittle_indices(
        states, P_active, P_passive, discount=discount
    )

    return states, indices


def lookup_whittle_index(
    knowledge: float,
    states: np.ndarray,
    indices: np.ndarray,
) -> float:
    """Look up Whittle index for a given knowledge value via nearest state."""
    idx = _nearest_state(states, knowledge)
    return float(indices[idx])


# ---------------------------------------------------------------------------
# Analytical approximation (for speed and closed-form insights)
# ---------------------------------------------------------------------------

def approximate_whittle_index(
    k: float,
    learning_rate: float = 0.12,
    incorrect_penalty: float = 0.02,
    decay_rate: float = 0.01,
    base_knowledge: float = 0.10,
    discount: float = 0.99,
) -> float:
    """
    Approximate Whittle index using one-step lookahead analysis.

    Computes the subsidy at which the one-step advantage of quizzing equals
    the one-step advantage of not quizzing (with subsidy).

    Active value: E[k'] = k·(k + α(1-k)) + (1-k)·(k - β·k) = k + α·k·(1-k) - β·k·(1-k)
                        = k + (α - β)·k·(1-k)  [for α > β, this is positive]
    Passive value: k' = base + (k - base)·exp(-λ)

    The index W(k) ≈ active_gain - passive_loss:
    W(k) = [expected_k_active - k] - [k_passive - k]
         = (α - β)·k·(1-k) - [(base + (k-base)·exp(-λ)) - k]
         = (α - β)·k·(1-k) + (k - base)·(1 - exp(-λ))
    """
    # Expected knowledge gain from quizzing
    active_gain = (learning_rate - incorrect_penalty) * k * (1 - k)

    # Knowledge loss from NOT quizzing (forgetting)
    passive_loss = (k - base_knowledge) * (1 - math.exp(-decay_rate))

    # Whittle index ≈ active gain + passive loss (opportunity cost of not quizzing)
    W = active_gain + passive_loss

    return W
