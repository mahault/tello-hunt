"""
Clone-structured Hidden Markov Model (CHMM) implementation.

Adapted from: https://github.com/vicariousinc/naturecomm_cscg

Key concepts:
- Multiple "clone" states can emit the same observation
- Resolves perceptual aliasing (e.g., two hallways look the same)
- Learns transition structure T(z'|z,a) and emission structure E(x|z)
"""

import numpy as np
import numba as nb
from typing import Tuple, Optional, List
from dataclasses import dataclass


@nb.njit(cache=True)
def forward(T: np.ndarray, x: np.ndarray, a: np.ndarray,
            pseudocount: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Forward message passing for CHMM.

    Args:
        T: Transition tensor (n_states, n_states, n_actions) - T[j,i,a] = P(z'=j|z=i, a)
        x: Observation sequence (seq_len,) - observation indices
        a: Action sequence (seq_len,) - action indices (a[t] is action taken before x[t])
        pseudocount: Smoothing for zero probabilities

    Returns:
        alpha: Forward messages (seq_len, n_states)
        log_lik: Log-likelihood of sequence
    """
    seq_len = len(x)
    n_states = T.shape[0]

    # Initialize with uniform
    alpha = np.zeros((seq_len, n_states), dtype=np.float64)
    alpha[0] = 1.0 / n_states

    log_lik = 0.0

    for t in range(1, seq_len):
        # Transition: alpha[t] = T[:,:,a[t]] @ alpha[t-1]
        action = a[t]
        for j in range(n_states):
            for i in range(n_states):
                alpha[t, j] += T[j, i, action] * alpha[t-1, i]

        # Normalize to prevent underflow
        norm = alpha[t].sum()
        if norm > 0:
            alpha[t] /= norm
            log_lik += np.log(norm)
        else:
            # Handle zero probability (novel observation)
            alpha[t] = 1.0 / n_states
            log_lik += np.log(1e-10)

    return alpha, log_lik


@nb.njit(cache=True)
def backward(T: np.ndarray, x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Backward message passing for CHMM.

    Args:
        T: Transition tensor (n_states, n_states, n_actions)
        x: Observation sequence (seq_len,)
        a: Action sequence (seq_len,)

    Returns:
        beta: Backward messages (seq_len, n_states)
    """
    seq_len = len(x)
    n_states = T.shape[0]

    beta = np.zeros((seq_len, n_states), dtype=np.float64)
    beta[-1] = 1.0

    for t in range(seq_len - 2, -1, -1):
        action = a[t + 1]
        for i in range(n_states):
            for j in range(n_states):
                beta[t, i] += T[j, i, action] * beta[t+1, j]

        # Normalize
        norm = beta[t].sum()
        if norm > 0:
            beta[t] /= norm

    return beta


@nb.njit(cache=True)
def forward_mp(T: np.ndarray, x: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Max-product forward pass for Viterbi decoding.

    Returns:
        max_vals: Max values at each step (seq_len, n_states)
        backpointers: Backpointers for traceback (seq_len, n_states)
    """
    seq_len = len(x)
    n_states = T.shape[0]

    max_vals = np.zeros((seq_len, n_states), dtype=np.float64)
    backpointers = np.zeros((seq_len, n_states), dtype=np.int32)

    # Initialize
    max_vals[0] = 1.0 / n_states

    for t in range(1, seq_len):
        action = a[t]
        for j in range(n_states):
            best_val = -np.inf
            best_i = 0
            for i in range(n_states):
                val = T[j, i, action] * max_vals[t-1, i]
                if val > best_val:
                    best_val = val
                    best_i = i
            max_vals[t, j] = best_val
            backpointers[t, j] = best_i

        # Normalize
        norm = max_vals[t].sum()
        if norm > 0:
            max_vals[t] /= norm

    return max_vals, backpointers


@nb.njit(cache=True)
def backtrace(max_vals: np.ndarray, backpointers: np.ndarray) -> np.ndarray:
    """
    Backtrace through Viterbi path.
    """
    seq_len = max_vals.shape[0]
    path = np.zeros(seq_len, dtype=np.int32)

    # Start from best final state
    path[-1] = np.argmax(max_vals[-1])

    # Trace back
    for t in range(seq_len - 2, -1, -1):
        path[t] = backpointers[t + 1, path[t + 1]]

    return path


@nb.njit(cache=True)
def update_counts(T_counts: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
                  T: np.ndarray, x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Accumulate transition counts for EM update.

    E-step: compute expected counts E[N(i->j|a)]
    """
    seq_len = len(x)
    n_states = T.shape[0]

    for t in range(seq_len - 1):
        action = a[t + 1]

        # Compute xi(t, i, j) = P(z_t=i, z_{t+1}=j | x, a)
        xi_sum = 0.0
        xi = np.zeros((n_states, n_states), dtype=np.float64)

        for i in range(n_states):
            for j in range(n_states):
                xi[i, j] = alpha[t, i] * T[j, i, action] * beta[t+1, j]
                xi_sum += xi[i, j]

        # Normalize and accumulate
        if xi_sum > 0:
            for i in range(n_states):
                for j in range(n_states):
                    T_counts[j, i, action] += xi[i, j] / xi_sum

    return T_counts


@dataclass
class CHMMConfig:
    """Configuration for CHMM."""
    n_clones: np.ndarray  # Number of clones per observation token
    n_actions: int
    pseudocount: float = 0.1
    dtype: np.dtype = np.float64


class CHMM:
    """
    Clone-structured Hidden Markov Model.

    The state space consists of (observation_token, clone_index) pairs.
    Multiple clone states can emit the same observation, allowing the model
    to disambiguate perceptually identical but topologically distinct locations.

    Attributes:
        n_clones: Array of clone counts per observation token
        n_obs: Number of observation tokens
        n_states: Total number of states (sum of clones)
        n_actions: Number of action types
        T: Transition tensor (n_states, n_states, n_actions)
        obs_to_states: Mapping from observation token to state indices
    """

    def __init__(
        self,
        n_clones: np.ndarray,
        n_actions: int,
        pseudocount: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize CHMM.

        Args:
            n_clones: Number of clone states per observation token
            n_actions: Number of action types (e.g., 7 for stay/fwd/back/left/right/strafe_left/strafe_right)
            pseudocount: Dirichlet prior for smoothing
            seed: Random seed for initialization
        """
        self.n_clones = np.array(n_clones, dtype=np.int32)
        self.n_obs = len(n_clones)
        self.n_states = int(np.sum(n_clones))
        self.n_actions = n_actions
        self.pseudocount = pseudocount
        self.rng = np.random.default_rng(seed)

        # Build observation -> state mapping
        self.obs_to_states = []
        state_idx = 0
        for obs, n_clone in enumerate(n_clones):
            states = list(range(state_idx, state_idx + n_clone))
            self.obs_to_states.append(states)
            state_idx += n_clone

        # State -> observation mapping
        self.state_to_obs = np.zeros(self.n_states, dtype=np.int32)
        for obs, states in enumerate(self.obs_to_states):
            for s in states:
                self.state_to_obs[s] = obs

        # Initialize transition matrix with random + prior
        self._init_transition_matrix()

        # Accumulated counts for learning
        self.T_counts = np.zeros((self.n_states, self.n_states, n_actions), dtype=np.float64)

    def _init_transition_matrix(self):
        """Initialize transition matrix with random values + Dirichlet prior."""
        self.T = self.rng.random((self.n_states, self.n_states, self.n_actions))
        self.T += self.pseudocount

        # Normalize: sum over destination states = 1
        for a in range(self.n_actions):
            for i in range(self.n_states):
                self.T[:, i, a] /= self.T[:, i, a].sum()

    def forward(self, x: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass with observation masking.

        Only states that can emit the observed token have non-zero probability.

        Args:
            x: Observation sequence (seq_len,) - observation token indices
            a: Action sequence (seq_len,) - action indices

        Returns:
            alpha: Forward messages (seq_len, n_states)
            log_lik: Log-likelihood of sequence
        """
        x = np.asarray(x, dtype=np.int32)
        a = np.asarray(a, dtype=np.int32)

        seq_len = len(x)
        alpha = np.zeros((seq_len, self.n_states), dtype=np.float64)
        log_lik = 0.0

        # Initialize: uniform over states that emit x[0]
        valid_states = self.obs_to_states[x[0]]
        alpha[0, valid_states] = 1.0 / len(valid_states)

        for t in range(1, seq_len):
            action = a[t]
            valid_states = self.obs_to_states[x[t]]

            # Transition
            for j in valid_states:
                for i in range(self.n_states):
                    alpha[t, j] += self.T[j, i, action] * alpha[t-1, i]

            # Normalize
            norm = alpha[t].sum()
            if norm > 0:
                alpha[t] /= norm
                log_lik += np.log(norm)
            else:
                # Novel observation - uniform over valid states
                alpha[t, valid_states] = 1.0 / len(valid_states)
                log_lik += np.log(1e-10)

        return alpha, log_lik

    def backward(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Backward pass with observation masking."""
        x = np.asarray(x, dtype=np.int32)
        a = np.asarray(a, dtype=np.int32)

        seq_len = len(x)
        beta = np.zeros((seq_len, self.n_states), dtype=np.float64)
        beta[-1] = 1.0

        for t in range(seq_len - 2, -1, -1):
            action = a[t + 1]
            valid_next = self.obs_to_states[x[t + 1]]

            for i in range(self.n_states):
                for j in valid_next:
                    beta[t, i] += self.T[j, i, action] * beta[t+1, j]

            norm = beta[t].sum()
            if norm > 0:
                beta[t] /= norm

        return beta

    def decode(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding - find most likely state sequence.

        Args:
            x: Observation sequence
            a: Action sequence

        Returns:
            path: Most likely state sequence
        """
        x = np.asarray(x, dtype=np.int32)
        a = np.asarray(a, dtype=np.int32)

        seq_len = len(x)
        max_vals = np.zeros((seq_len, self.n_states), dtype=np.float64)
        backpointers = np.zeros((seq_len, self.n_states), dtype=np.int32)

        # Initialize
        valid_states = self.obs_to_states[x[0]]
        max_vals[0, valid_states] = 1.0 / len(valid_states)

        for t in range(1, seq_len):
            action = a[t]
            valid_states = self.obs_to_states[x[t]]

            for j in valid_states:
                best_val = -np.inf
                best_i = 0
                for i in range(self.n_states):
                    val = self.T[j, i, action] * max_vals[t-1, i]
                    if val > best_val:
                        best_val = val
                        best_i = i
                max_vals[t, j] = best_val
                backpointers[t, j] = best_i

            norm = max_vals[t].sum()
            if norm > 0:
                max_vals[t] /= norm

        # Backtrace
        path = np.zeros(seq_len, dtype=np.int32)
        path[-1] = np.argmax(max_vals[-1])

        for t in range(seq_len - 2, -1, -1):
            path[t] = backpointers[t + 1, path[t + 1]]

        return path

    def learn_em_T(
        self,
        x: np.ndarray,
        a: np.ndarray,
        n_iter: int = 10,
        reset_counts: bool = True
    ) -> List[float]:
        """
        Learn transition matrix via EM.

        Args:
            x: Observation sequence or list of sequences
            a: Action sequence or list of sequences
            n_iter: Number of EM iterations
            reset_counts: Whether to reset accumulated counts

        Returns:
            log_liks: Log-likelihood at each iteration
        """
        # Handle single sequence vs batch
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = [x]
            a = [a]

        log_liks = []

        if reset_counts:
            self.T_counts = np.zeros_like(self.T_counts)

        for iteration in range(n_iter):
            # E-step: accumulate expected counts
            total_log_lik = 0.0

            for xi, ai in zip(x, a):
                xi = np.asarray(xi, dtype=np.int32)
                ai = np.asarray(ai, dtype=np.int32)

                alpha, log_lik = self.forward(xi, ai)
                beta = self.backward(xi, ai)
                total_log_lik += log_lik

                self._accumulate_counts(alpha, beta, xi, ai)

            log_liks.append(total_log_lik)

            # M-step: update transition matrix
            self._update_T()

        return log_liks

    def _accumulate_counts(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        x: np.ndarray,
        a: np.ndarray
    ):
        """Accumulate transition counts (E-step)."""
        seq_len = len(x)

        for t in range(seq_len - 1):
            action = a[t + 1]
            valid_next = self.obs_to_states[x[t + 1]]

            # Compute xi(t, i, j) = P(z_t=i, z_{t+1}=j | x, a)
            xi = np.zeros((self.n_states, self.n_states), dtype=np.float64)

            for i in range(self.n_states):
                for j in valid_next:
                    xi[i, j] = alpha[t, i] * self.T[j, i, action] * beta[t+1, j]

            xi_sum = xi.sum()
            if xi_sum > 0:
                xi /= xi_sum
                self.T_counts[:, :, action] += xi.T  # Note: T[j,i,a] = P(j|i,a)

    def _update_T(self):
        """Update transition matrix from accumulated counts (M-step)."""
        # Add pseudocount and normalize
        for a in range(self.n_actions):
            T_a = self.T_counts[:, :, a] + self.pseudocount

            for i in range(self.n_states):
                col_sum = T_a[:, i].sum()
                if col_sum > 0:
                    self.T[:, i, a] = T_a[:, i] / col_sum

    def belief(self, x: int, prev_belief: Optional[np.ndarray] = None,
               action: int = 0) -> np.ndarray:
        """
        Single-step belief update.

        Args:
            x: Current observation token
            prev_belief: Previous belief over states (None = uniform)
            action: Action taken to reach current observation

        Returns:
            belief: Updated belief over states
        """
        if prev_belief is None:
            # Initialize uniform over states that emit x
            belief = np.zeros(self.n_states, dtype=np.float64)
            valid_states = self.obs_to_states[x]
            belief[valid_states] = 1.0 / len(valid_states)
            return belief

        # Transition
        belief = self.T[:, :, action] @ prev_belief

        # Observation masking - zero out states that can't emit x
        valid_states = self.obs_to_states[x]
        mask = np.zeros(self.n_states, dtype=np.float64)
        mask[valid_states] = 1.0
        belief *= mask

        # Normalize
        norm = belief.sum()
        if norm > 0:
            belief /= norm
        else:
            # Novel - uniform over valid states
            belief[valid_states] = 1.0 / len(valid_states)

        return belief

    def bridge(
        self,
        start_state: int,
        end_state: int,
        max_steps: int = 100
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find action sequence to navigate between states (path planning).

        Uses BFS on the transition graph.

        Args:
            start_state: Starting state index
            end_state: Target state index
            max_steps: Maximum path length

        Returns:
            path: List of (state, action) tuples, or None if no path found
        """
        from collections import deque

        # BFS
        queue = deque([(start_state, [])])
        visited = {start_state}

        while queue:
            state, path = queue.popleft()

            if state == end_state:
                return path

            if len(path) >= max_steps:
                continue

            # Try each action
            for action in range(self.n_actions):
                # Find most likely next state
                probs = self.T[:, state, action]
                next_state = np.argmax(probs)

                if next_state not in visited and probs[next_state] > 0.01:
                    visited.add(next_state)
                    queue.append((next_state, path + [(state, action)]))

        return None

    def get_transition_graph(self, threshold: float = 0.1) -> List[Tuple[int, int, int, float]]:
        """
        Extract transition graph edges.

        Returns:
            edges: List of (from_state, to_state, action, probability)
        """
        edges = []
        for a in range(self.n_actions):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    if self.T[j, i, a] > threshold:
                        edges.append((i, j, a, self.T[j, i, a]))
        return edges

    def save(self, filepath: str):
        """Save model to file."""
        np.savez(
            filepath,
            n_clones=self.n_clones,
            n_actions=self.n_actions,
            pseudocount=self.pseudocount,
            T=self.T,
            T_counts=self.T_counts
        )

    @classmethod
    def load(cls, filepath: str) -> 'CHMM':
        """Load model from file."""
        data = np.load(filepath)
        model = cls(
            n_clones=data['n_clones'],
            n_actions=int(data['n_actions']),
            pseudocount=float(data['pseudocount'])
        )
        model.T = data['T']
        model.T_counts = data['T_counts']
        return model

    def __repr__(self) -> str:
        return (f"CHMM(n_obs={self.n_obs}, n_states={self.n_states}, "
                f"n_actions={self.n_actions}, clones={self.n_clones.tolist()})")
