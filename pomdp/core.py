"""
Core JAX functions for POMDP belief updates and action selection.

All functions are JIT-compiled for real-time performance.
Uses active inference framework with Expected Free Energy for action selection.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from .config import ACTION_TEMPERATURE, EPISTEMIC_WEIGHT, PRAGMATIC_WEIGHT


# =============================================================================
# Basic Operations
# =============================================================================

@jax.jit
def normalize(x: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
    """Normalize array to sum to 1 (probability distribution)."""
    return x / (jnp.sum(x) + eps)


@jax.jit
def log_normalize(log_x: jnp.ndarray) -> jnp.ndarray:
    """Normalize in log space (log-sum-exp trick)."""
    max_log = jnp.max(log_x)
    return log_x - max_log - jnp.log(jnp.sum(jnp.exp(log_x - max_log)))


@jax.jit
def softmax(x: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """Softmax with temperature parameter."""
    x_scaled = x / temperature
    x_max = jnp.max(x_scaled)
    exp_x = jnp.exp(x_scaled - x_max)
    return exp_x / (jnp.sum(exp_x) + 1e-10)


@jax.jit
def entropy(p: jnp.ndarray, eps: float = 1e-10) -> float:
    """Compute entropy of probability distribution."""
    return -jnp.sum(p * jnp.log(p + eps))


@jax.jit
def kl_divergence(p: jnp.ndarray, q: jnp.ndarray, eps: float = 1e-10) -> float:
    """Compute KL divergence D_KL(p || q)."""
    return jnp.sum(p * (jnp.log(p + eps) - jnp.log(q + eps)))


# =============================================================================
# Belief Update Functions
# =============================================================================

@jax.jit
def belief_update(
    prior: jnp.ndarray,
    likelihood: jnp.ndarray,
    eps: float = 1e-10
) -> jnp.ndarray:
    """
    Bayesian belief update.

    posterior = P(s | o) ∝ P(o | s) * P(s)

    Args:
        prior: P(s) - prior belief over states, shape (n_states,)
        likelihood: P(o | s) - likelihood of observation given each state, shape (n_states,)

    Returns:
        posterior: P(s | o) - posterior belief over states, shape (n_states,)
    """
    unnormalized = likelihood * prior
    return normalize(unnormalized, eps)


@jax.jit
def belief_update_from_A(
    prior: jnp.ndarray,
    A: jnp.ndarray,
    obs_idx: int,
    eps: float = 1e-10
) -> jnp.ndarray:
    """
    Belief update using A matrix (observation model).

    Args:
        prior: P(s) - prior belief over states, shape (n_states,)
        A: P(o | s) - observation model, shape (n_obs, n_states)
        obs_idx: Index of observed observation

    Returns:
        posterior: P(s | o) - posterior belief, shape (n_states,)
    """
    likelihood = A[obs_idx, :]
    return belief_update(prior, likelihood, eps)


@jax.jit
def belief_update_multi_modality(
    prior: jnp.ndarray,
    A_list: Tuple[jnp.ndarray, ...],
    obs_indices: jnp.ndarray,
    eps: float = 1e-10
) -> jnp.ndarray:
    """
    Belief update with multiple observation modalities.

    Multiplies likelihoods from all modalities.

    Args:
        prior: P(s) - prior belief, shape (n_states,)
        A_list: Tuple of A matrices, each shape (n_obs_i, n_states)
        obs_indices: Observed indices for each modality, shape (n_modalities,)

    Returns:
        posterior: P(s | o1, o2, ...) - posterior belief
    """
    combined_likelihood = jnp.ones_like(prior)

    for i, A in enumerate(A_list):
        obs_idx = obs_indices[i]
        likelihood = A[obs_idx, :]
        combined_likelihood = combined_likelihood * likelihood

    return belief_update(prior, combined_likelihood, eps)


# =============================================================================
# Variational Free Energy (VFE) - for perception/inference
# =============================================================================

@jax.jit
def accuracy(
    belief: jnp.ndarray,
    A: jnp.ndarray,
    obs_idx: int,
    eps: float = 1e-10
) -> float:
    """
    Compute accuracy term of VFE: E_q[log p(o|s)]

    How well the current belief explains the observation.
    Higher = better fit.

    Args:
        belief: Current belief q(s), shape (n_states,)
        A: Observation model P(o|s), shape (n_obs, n_states)
        obs_idx: Observed observation index

    Returns:
        accuracy: E_q[log p(o|s)] (scalar, higher = better)
    """
    # P(o|s) for the observed o
    likelihood = A[obs_idx, :]  # shape (n_states,)

    # E_q[log p(o|s)] = sum_s q(s) * log p(o|s)
    return jnp.sum(belief * jnp.log(likelihood + eps))


@jax.jit
def complexity(
    posterior: jnp.ndarray,
    prior: jnp.ndarray,
    eps: float = 1e-10
) -> float:
    """
    Compute complexity term of VFE: KL[q(s) || p(s)]

    How much the posterior diverged from the prior.
    Lower = simpler update (less surprise).

    Args:
        posterior: Posterior belief q(s), shape (n_states,)
        prior: Prior belief p(s), shape (n_states,)

    Returns:
        complexity: KL divergence (scalar, lower = simpler)
    """
    return kl_divergence(posterior, prior, eps)


@jax.jit
def variational_free_energy(
    posterior: jnp.ndarray,
    prior: jnp.ndarray,
    A: jnp.ndarray,
    obs_idx: int,
    eps: float = 1e-10
) -> float:
    """
    Compute Variational Free Energy (VFE).

    VFE = complexity - accuracy
        = KL[q(s) || p(s)] - E_q[log p(o|s)]

    Lower VFE = better model fit.
    High VFE = observation doesn't fit model well (novelty/surprise).

    Args:
        posterior: Posterior belief q(s) after update, shape (n_states,)
        prior: Prior belief p(s) before update, shape (n_states,)
        A: Observation model P(o|s), shape (n_obs, n_states)
        obs_idx: Observed observation index

    Returns:
        F: Variational free energy (scalar, lower = better fit)
    """
    acc = accuracy(posterior, A, obs_idx, eps)
    comp = complexity(posterior, prior, eps)

    return comp - acc


@jax.jit
def surprisal(
    A: jnp.ndarray,
    prior: jnp.ndarray,
    obs_idx: int,
    eps: float = 1e-10
) -> float:
    """
    Compute surprisal (negative log evidence): -log p(o)

    p(o) = sum_s p(o|s) p(s)

    High surprisal = unexpected observation (novelty).
    This is an upper bound on VFE.

    Args:
        A: Observation model P(o|s), shape (n_obs, n_states)
        prior: Prior belief p(s), shape (n_states,)
        obs_idx: Observed observation index

    Returns:
        surprisal: -log p(o) (scalar, higher = more surprising)
    """
    # P(o|s) for observed o
    likelihood = A[obs_idx, :]  # shape (n_states,)

    # P(o) = sum_s P(o|s) P(s)
    marginal_likelihood = jnp.sum(likelihood * prior)

    return -jnp.log(marginal_likelihood + eps)


@jax.jit
def vfe_components(
    posterior: jnp.ndarray,
    prior: jnp.ndarray,
    A: jnp.ndarray,
    obs_idx: int,
    eps: float = 1e-10
) -> Tuple[float, float, float, float]:
    """
    Compute all VFE-related quantities at once.

    Returns:
        vfe: Variational free energy
        acc: Accuracy term (higher = better)
        comp: Complexity term (lower = simpler)
        surp: Surprisal (higher = more novel)
    """
    acc = accuracy(posterior, A, obs_idx, eps)
    comp = complexity(posterior, prior, eps)
    vfe = comp - acc
    surp = surprisal(A, prior, obs_idx, eps)

    return vfe, acc, comp, surp


@jax.jit
def is_novel_observation(
    A: jnp.ndarray,
    prior: jnp.ndarray,
    obs_idx: int,
    novelty_threshold: float = 5.0,
    eps: float = 1e-10
) -> bool:
    """
    Check if observation is novel (high surprisal).

    Args:
        A: Observation model P(o|s), shape (n_obs, n_states)
        prior: Prior belief p(s), shape (n_states,)
        obs_idx: Observed observation index
        novelty_threshold: Surprisal threshold for novelty

    Returns:
        is_novel: True if observation is surprising
    """
    surp = surprisal(A, prior, obs_idx, eps)
    return surp > novelty_threshold


@jax.jit
def belief_update_with_vfe(
    prior: jnp.ndarray,
    A: jnp.ndarray,
    obs_idx: int,
    eps: float = 1e-10
) -> Tuple[jnp.ndarray, float, float, float, float]:
    """
    Perform belief update and compute VFE components.

    Combines perception (belief update) with VFE monitoring.

    Args:
        prior: Prior belief p(s), shape (n_states,)
        A: Observation model P(o|s), shape (n_obs, n_states)
        obs_idx: Observed observation index

    Returns:
        posterior: Updated belief q(s)
        vfe: Variational free energy
        accuracy: Accuracy term
        complexity: Complexity term
        surprisal: Surprisal (-log evidence)
    """
    # Belief update
    likelihood = A[obs_idx, :]
    posterior = belief_update(prior, likelihood, eps)

    # VFE components
    vfe, acc, comp, surp = vfe_components(posterior, prior, A, obs_idx, eps)

    return posterior, vfe, acc, comp, surp


# =============================================================================
# State Prediction
# =============================================================================

@jax.jit
def predict_next_belief(
    belief: jnp.ndarray,
    B: jnp.ndarray,
    action_idx: int = 0
) -> jnp.ndarray:
    """
    Predict belief after taking action (using transition model).

    P(s') = Σ_s P(s' | s, a) * P(s)

    Args:
        belief: Current belief P(s), shape (n_states,)
        B: Transition model, shape (n_states, n_states) or (n_states, n_states, n_actions)
        action_idx: Action index if B has action dimension

    Returns:
        predicted: Predicted belief P(s'), shape (n_states,)
    """
    if B.ndim == 3:
        transition = B[:, :, action_idx]  # (n_states, n_states)
    else:
        transition = B

    # B[s', s] = P(s' | s), so we need B @ belief
    predicted = jnp.dot(transition, belief)
    return normalize(predicted)


@jax.jit
def predict_observation(
    belief: jnp.ndarray,
    A: jnp.ndarray
) -> jnp.ndarray:
    """
    Predict expected observation distribution given belief.

    P(o) = Σ_s P(o | s) * P(s)

    Args:
        belief: Current belief P(s), shape (n_states,)
        A: Observation model P(o | s), shape (n_obs, n_states)

    Returns:
        expected_obs: Expected observation distribution P(o), shape (n_obs,)
    """
    return jnp.dot(A, belief)


# =============================================================================
# Expected Free Energy (for Active Inference)
# =============================================================================

@jax.jit
def expected_free_energy(
    belief: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    action_idx: int
) -> float:
    """
    Compute Expected Free Energy (EFE) for an action.

    G = ambiguity - pragmatic_value

    Where:
    - Ambiguity: Expected entropy of observations (epistemic term)
    - Pragmatic value: Expected log probability of preferred observations

    Lower G = better action.

    Args:
        belief: Current belief P(s), shape (n_states,)
        A: Observation model P(o | s), shape (n_obs, n_states)
        B: Transition model P(s' | s, a), shape (n_states, n_states, n_actions)
        C: Observation preferences (higher = more preferred), shape (n_obs,)
        action_idx: Action to evaluate

    Returns:
        G: Expected free energy (scalar)
    """
    # Predict next belief under action
    predicted_belief = predict_next_belief(belief, B, action_idx)

    # Expected observation distribution
    expected_obs = predict_observation(predicted_belief, A)

    # Epistemic value: negative entropy (want to reduce uncertainty)
    # High entropy = high ambiguity = bad
    ambiguity = entropy(expected_obs)

    # Pragmatic value: expected preference satisfaction
    # C should be log probabilities of preferred observations
    log_C = jnp.log(C + 1e-10)
    pragmatic = jnp.dot(expected_obs, log_C)

    # EFE: ambiguity - pragmatic (lower is better)
    G = EPISTEMIC_WEIGHT * ambiguity - PRAGMATIC_WEIGHT * pragmatic

    return G


@jax.jit
def compute_all_efe(
    belief: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    n_actions: int
) -> jnp.ndarray:
    """
    Compute EFE for all actions.

    Args:
        belief: Current belief P(s)
        A: Observation model
        B: Transition model with action dimension
        C: Observation preferences
        n_actions: Number of actions

    Returns:
        G_values: EFE for each action, shape (n_actions,)
    """
    G_values = jnp.array([
        expected_free_energy(belief, A, B, C, a)
        for a in range(n_actions)
    ])
    return G_values


# =============================================================================
# Action Selection
# =============================================================================

@jax.jit
def select_action(
    belief: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    n_actions: int,
    temperature: float = ACTION_TEMPERATURE
) -> Tuple[int, jnp.ndarray]:
    """
    Select action by minimizing Expected Free Energy.

    Args:
        belief: Current belief P(s)
        A: Observation model
        B: Transition model
        C: Observation preferences
        n_actions: Number of available actions
        temperature: Softmax temperature (lower = more deterministic)

    Returns:
        action_idx: Selected action index
        action_probs: Probability distribution over actions
    """
    # Compute EFE for all actions
    G_values = compute_all_efe(belief, A, B, C, n_actions)

    # Convert to action probabilities (lower G = higher prob)
    # Use negative G since softmax picks highest
    action_probs = softmax(-G_values, temperature)

    # Return argmin (lowest EFE)
    action_idx = jnp.argmin(G_values)

    return action_idx, action_probs


# =============================================================================
# Dirichlet-Categorical Learning
# =============================================================================

@jax.jit
def update_dirichlet_counts(
    counts: jnp.ndarray,
    obs_idx: int,
    state_idx: int,
    learning_rate: float = 1.0
) -> jnp.ndarray:
    """
    Update Dirichlet counts for online learning.

    Args:
        counts: Current counts, shape (n_obs, n_states)
        obs_idx: Observed observation index
        state_idx: Current state index
        learning_rate: How much to increment (default 1.0)

    Returns:
        updated_counts: Updated counts
    """
    return counts.at[obs_idx, state_idx].add(learning_rate)


@jax.jit
def counts_to_distribution(
    counts: jnp.ndarray,
    prior_alpha: float = 1.0
) -> jnp.ndarray:
    """
    Convert Dirichlet counts to probability distribution.

    P(o | s) = (counts[o, s] + alpha) / sum_o(counts[o, s] + alpha)

    Args:
        counts: Observation counts, shape (n_obs, n_states)
        prior_alpha: Dirichlet prior concentration

    Returns:
        A: Normalized probability matrix, shape (n_obs, n_states)
    """
    # Add prior and normalize each column
    counts_with_prior = counts + prior_alpha
    col_sums = jnp.sum(counts_with_prior, axis=0, keepdims=True)
    return counts_with_prior / (col_sums + 1e-10)


# =============================================================================
# Utility Functions for Dynamic State Spaces
# =============================================================================

def expand_belief(belief: jnp.ndarray, new_size: int) -> jnp.ndarray:
    """
    Expand belief vector to accommodate new states.

    New states get small uniform probability mass.

    Args:
        belief: Current belief, shape (current_size,)
        new_size: Target size (must be >= current_size)

    Returns:
        expanded: Expanded belief, shape (new_size,)
    """
    current_size = belief.shape[0]
    if new_size <= current_size:
        return belief

    # Allocate small probability to new states
    new_prob_mass = 0.01  # Small mass for new states
    n_new = new_size - current_size

    # Renormalize existing beliefs
    old_total = 1.0 - new_prob_mass
    scaled_old = belief * old_total

    # New states get uniform small probability
    new_probs = jnp.ones(n_new) * (new_prob_mass / n_new)

    return jnp.concatenate([scaled_old, new_probs])


def expand_A_matrix(
    A: jnp.ndarray,
    new_n_states: int,
    prior_alpha: float = 1.0
) -> jnp.ndarray:
    """
    Expand A matrix to accommodate new states.

    New columns get uniform prior.

    Args:
        A: Current A matrix, shape (n_obs, n_states)
        new_n_states: Target number of states
        prior_alpha: Prior for new columns

    Returns:
        expanded: Expanded A matrix, shape (n_obs, new_n_states)
    """
    n_obs, current_n_states = A.shape
    if new_n_states <= current_n_states:
        return A

    n_new = new_n_states - current_n_states

    # New columns: uniform distribution
    new_cols = jnp.ones((n_obs, n_new)) / n_obs

    return jnp.concatenate([A, new_cols], axis=1)


def expand_B_matrix(
    B: jnp.ndarray,
    new_n_states: int
) -> jnp.ndarray:
    """
    Expand B transition matrix for new states.

    New states have self-loop transitions.

    Args:
        B: Current B matrix, shape (n_states, n_states) or (n_states, n_states, n_actions)
        new_n_states: Target number of states

    Returns:
        expanded: Expanded B matrix
    """
    if B.ndim == 2:
        current_n = B.shape[0]
        if new_n_states <= current_n:
            return B

        n_new = new_n_states - current_n

        # Expand rows and columns
        # New states mostly self-loop
        new_B = jnp.eye(new_n_states) * 0.9
        new_B = new_B + jnp.ones((new_n_states, new_n_states)) * (0.1 / new_n_states)

        # Copy old transitions
        new_B = new_B.at[:current_n, :current_n].set(B)

        return new_B

    elif B.ndim == 3:
        current_n, _, n_actions = B.shape
        if new_n_states <= current_n:
            return B

        # Expand each action's transition matrix
        new_B = jnp.zeros((new_n_states, new_n_states, n_actions))

        for a in range(n_actions):
            # Default: mostly self-loop
            new_B = new_B.at[:, :, a].set(
                jnp.eye(new_n_states) * 0.9 +
                jnp.ones((new_n_states, new_n_states)) * (0.1 / new_n_states)
            )
            # Copy old transitions
            new_B = new_B.at[:current_n, :current_n, a].set(B[:, :, a])

        return new_B

    return B
