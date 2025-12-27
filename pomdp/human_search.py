"""
Human Search POMDP for tracking belief over person locations.

This module maintains:
- Belief over where humans are located in the learned map
- Prior over human locations (learned from sightings)
- Search target recommendations based on belief

State space: N_locations + 1 (where +1 = "not visible / not in any known location")
Observation space: Person detection categories (not_detected, left, center, right, close)
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

from .config import (
    N_MAX_LOCATIONS,
    N_PERSON_OBS,
    PERSON_OBS,
    DIRICHLET_PRIOR_ALPHA,
    EPISTEMIC_WEIGHT,
    PRAGMATIC_WEIGHT,
)
from .core import (
    normalize,
    softmax,
    entropy,
    belief_update,
    predict_observation,
)
from .observation_encoder import ObservationToken


# Human location states: each learned location + "not visible"
# Index N_locations = human is not at any known location
NOT_VISIBLE_STATE = -1  # Will be mapped to last index dynamically


@dataclass
class HumanSearchResult:
    """Result of human search belief update."""
    belief: jnp.ndarray  # Belief over human locations (n_locations + 1,)
    most_likely_location: int  # Most probable location (-1 = not visible)
    confidence: float  # Max probability
    person_detected: bool  # Whether person was detected this frame
    search_target: int  # Recommended location to search next
    belief_entropy: float  # Uncertainty over human location


class HumanSearchPOMDP:
    """
    POMDP for tracking where humans are likely to be.

    State: Human's location (one of N learned locations, or "not visible")
    Observation: Person detection from YOLO (not_detected, left, center, right, close)

    Key insight: We only see the person if:
    1. Human is at same location as drone
    2. Human is actually looking at camera (close detection)

    A matrix: P(person_obs | human_loc, drone_loc)
    - If human_loc == drone_loc: high prob of detection
    - If human_loc != drone_loc: high prob of not_detected
    """

    def __init__(
        self,
        n_locations: int = 1,
        prior_alpha: float = DIRICHLET_PRIOR_ALPHA,
    ):
        """
        Initialize human search POMDP.

        Args:
            n_locations: Initial number of known locations
            prior_alpha: Dirichlet prior for learning
        """
        self.n_locations = n_locations
        self.prior_alpha = prior_alpha

        # Belief over human location (n_locations + 1 for "not visible")
        # Start with uniform prior
        self._belief: Optional[jnp.ndarray] = None

        # Learned prior: where humans have been seen before
        # Dirichlet counts for each location
        self._human_sighting_counts = np.zeros(N_MAX_LOCATIONS + 1)
        self._human_sighting_counts[-1] = 1.0  # Small prior for "not visible"

        # Last known location where person was detected
        self._last_sighting_location: Optional[int] = None
        self._frames_since_sighting: int = 0

    @property
    def belief(self) -> jnp.ndarray:
        """Current belief over human locations."""
        if self._belief is None:
            n_states = self.n_locations + 1
            self._belief = jnp.ones(n_states) / n_states
        return self._belief

    @property
    def n_states(self) -> int:
        """Number of human location states (locations + not_visible)."""
        return self.n_locations + 1

    def expand_to_locations(self, new_n_locations: int) -> None:
        """
        Expand state space when new locations are discovered.

        Args:
            new_n_locations: New number of known locations
        """
        if new_n_locations <= self.n_locations:
            return

        old_n_states = self.n_states
        self.n_locations = new_n_locations
        new_n_states = self.n_states

        if self._belief is not None:
            # Expand belief - move "not visible" to end
            old_belief = np.array(self._belief)
            new_belief = np.zeros(new_n_states)

            # Copy location beliefs (excluding old "not visible")
            new_belief[:old_n_states - 1] = old_belief[:old_n_states - 1]

            # Add small probability for new locations
            n_new = new_n_locations - (old_n_states - 1)
            new_prob = 0.01 / n_new
            for i in range(old_n_states - 1, new_n_states - 1):
                new_belief[i] = new_prob

            # "Not visible" stays at end with remaining probability
            new_belief[-1] = old_belief[-1]

            # Renormalize
            self._belief = jnp.array(new_belief)
            self._belief = normalize(self._belief)

    def _build_A_matrix(self, drone_location: int) -> jnp.ndarray:
        """
        Build observation likelihood matrix P(person_obs | human_loc).

        Args:
            drone_location: Current drone location

        Returns:
            A: Shape (N_PERSON_OBS, n_states)
        """
        n_states = self.n_states
        A = np.zeros((N_PERSON_OBS, n_states))

        # Observation indices
        NOT_DETECTED = 0
        DETECTED_LEFT = 1
        DETECTED_CENTER = 2
        DETECTED_RIGHT = 3
        DETECTED_CLOSE = 4

        for human_loc in range(n_states):
            if human_loc == n_states - 1:
                # Human is "not visible" - very unlikely to detect
                A[NOT_DETECTED, human_loc] = 0.95
                A[DETECTED_LEFT, human_loc] = 0.01
                A[DETECTED_CENTER, human_loc] = 0.02
                A[DETECTED_RIGHT, human_loc] = 0.01
                A[DETECTED_CLOSE, human_loc] = 0.01

            elif human_loc == drone_location:
                # Human at same location as drone - likely to detect
                A[NOT_DETECTED, human_loc] = 0.10
                A[DETECTED_LEFT, human_loc] = 0.20
                A[DETECTED_CENTER, human_loc] = 0.30
                A[DETECTED_RIGHT, human_loc] = 0.20
                A[DETECTED_CLOSE, human_loc] = 0.20

            else:
                # Human at different location - unlikely to detect
                A[NOT_DETECTED, human_loc] = 0.90
                A[DETECTED_LEFT, human_loc] = 0.025
                A[DETECTED_CENTER, human_loc] = 0.025
                A[DETECTED_RIGHT, human_loc] = 0.025
                A[DETECTED_CLOSE, human_loc] = 0.025

        return jnp.array(A)

    def _build_B_matrix(self) -> jnp.ndarray:
        """
        Build transition matrix P(human_loc' | human_loc).

        Humans are mostly stationary but can move.

        Returns:
            B: Shape (n_states, n_states)
        """
        n_states = self.n_states

        # Mostly stay in place
        stay_prob = 0.85
        move_prob = (1.0 - stay_prob) / (n_states - 1)

        B = np.ones((n_states, n_states)) * move_prob
        np.fill_diagonal(B, stay_prob)

        # "Not visible" is more sticky (human doesn't suddenly appear)
        B[-1, :] = move_prob * 0.5
        B[-1, -1] = 0.95

        # But can transition to "not visible" (human leaves)
        B[:, -1] = 0.02
        B[-1, -1] = 0.95

        # Renormalize columns
        B = B / B.sum(axis=0, keepdims=True)

        return jnp.array(B)

    def _get_learned_prior(self) -> jnp.ndarray:
        """
        Get learned prior over human locations from sighting history.

        Returns:
            prior: Normalized prior, shape (n_states,)
        """
        n_states = self.n_states
        counts = self._human_sighting_counts[:n_states].copy()

        # Add Dirichlet prior
        counts += self.prior_alpha

        return jnp.array(counts / counts.sum())

    def _person_obs_from_token(self, obs: ObservationToken) -> int:
        """
        Convert observation token to person observation index.

        Args:
            obs: Observation token from YOLO

        Returns:
            obs_idx: Index into PERSON_OBS
        """
        # Use the pre-computed person_obs_idx if available
        if hasattr(obs, 'person_obs_idx'):
            return obs.person_obs_idx

        # Fallback to computing from individual fields
        if not obs.person_detected:
            return 0  # not_detected

        # Check if close first
        if obs.person_area > 0.5:
            return 4  # detected_close

        # Check position
        cx = obs.person_cx
        if cx < -0.3:
            return 1  # detected_left
        elif cx > 0.3:
            return 3  # detected_right
        else:
            return 2  # detected_center

    def update(
        self,
        obs: ObservationToken,
        drone_location: int,
    ) -> HumanSearchResult:
        """
        Update belief over human locations based on observation.

        Args:
            obs: Current observation token (with person detection)
            drone_location: Current drone location in map

        Returns:
            result: HumanSearchResult with updated belief and recommendations
        """
        # Ensure state space is large enough
        if drone_location >= self.n_locations:
            self.expand_to_locations(drone_location + 1)

        # Get person observation
        person_obs_idx = self._person_obs_from_token(obs)
        person_detected = person_obs_idx > 0

        # Build A matrix for current drone location
        A = self._build_A_matrix(drone_location)

        # Predict (human might have moved)
        B = self._build_B_matrix()
        predicted_belief = jnp.dot(B, self.belief)
        predicted_belief = normalize(predicted_belief)

        # Update belief based on observation
        likelihood = A[person_obs_idx, :]
        posterior = belief_update(predicted_belief, likelihood)

        self._belief = posterior

        # Track sightings for prior learning
        if person_detected:
            self._last_sighting_location = drone_location
            self._frames_since_sighting = 0
            self._update_sighting_counts(drone_location)
        else:
            self._frames_since_sighting += 1

        # Find most likely location
        most_likely_idx = int(jnp.argmax(posterior))
        confidence = float(posterior[most_likely_idx])

        # Map back to location (-1 for not visible)
        most_likely_location = most_likely_idx if most_likely_idx < self.n_locations else -1

        # Get search target
        search_target = self._get_search_target(drone_location)

        # Compute entropy
        belief_entropy = float(entropy(posterior))

        return HumanSearchResult(
            belief=posterior,
            most_likely_location=most_likely_location,
            confidence=confidence,
            person_detected=person_detected,
            search_target=search_target,
            belief_entropy=belief_entropy,
        )

    def _update_sighting_counts(self, location: int) -> None:
        """Update sighting counts for learning prior."""
        if 0 <= location < N_MAX_LOCATIONS:
            self._human_sighting_counts[location] += 1.0

    def _get_search_target(self, current_location: int) -> int:
        """
        Get recommended location to search next.

        Balances:
        - High belief locations (go where human likely is)
        - High prior locations (go where humans have been seen before)
        - Exploration (reduce uncertainty)

        Args:
            current_location: Current drone location

        Returns:
            target_location: Recommended location to go to (-1 if should stay)
        """
        belief = np.array(self.belief)
        n_locs = self.n_locations

        # If very confident human is here, stay
        if current_location < n_locs and belief[current_location] > 0.5:
            return current_location

        # Combine belief with learned prior
        prior = np.array(self._get_learned_prior())

        # Score for each location (excluding "not visible")
        scores = np.zeros(n_locs)
        for loc in range(n_locs):
            # Belief component
            belief_score = belief[loc]

            # Prior component (historical sightings)
            prior_score = prior[loc] * 0.3

            # Don't recommend current location unless very high belief
            if loc == current_location:
                scores[loc] = belief_score * 0.5 + prior_score
            else:
                scores[loc] = belief_score + prior_score

        # Return highest scoring location
        if len(scores) > 0:
            return int(np.argmax(scores))

        return 0

    def get_location_belief(self, location: int) -> float:
        """Get belief that human is at specific location."""
        if location < 0 or location >= self.n_states:
            return 0.0
        return float(self.belief[location])

    def get_not_visible_belief(self) -> float:
        """Get belief that human is not at any known location."""
        return float(self.belief[-1])

    def reset_belief_uniform(self) -> None:
        """Reset to uniform belief over all locations."""
        n_states = self.n_states
        self._belief = jnp.ones(n_states) / n_states

    def reset_belief_to_prior(self) -> None:
        """Reset belief to learned prior."""
        self._belief = self._get_learned_prior()

    def set_belief_at_location(self, location: int, confidence: float = 0.8) -> None:
        """
        Set high belief that human is at specific location.

        Useful when human is detected - concentrate belief there.

        Args:
            location: Location index
            confidence: Probability mass to put at location
        """
        n_states = self.n_states
        remaining = (1.0 - confidence) / (n_states - 1)

        belief = np.ones(n_states) * remaining
        if 0 <= location < n_states:
            belief[location] = confidence

        self._belief = jnp.array(belief)

    def get_statistics(self) -> Dict:
        """Get summary statistics for debugging."""
        belief = np.array(self.belief)
        n_locs = self.n_locations

        return {
            'n_locations': n_locs,
            'total_sightings': float(self._human_sighting_counts[:n_locs].sum()),
            'last_sighting_location': self._last_sighting_location,
            'frames_since_sighting': self._frames_since_sighting,
            'belief_entropy': float(entropy(self.belief)),
            'max_belief': float(belief.max()),
            'max_belief_location': int(np.argmax(belief[:n_locs])) if n_locs > 0 else -1,
            'not_visible_belief': float(belief[-1]),
        }

    def save_state(self) -> Dict:
        """Save state for persistence."""
        return {
            'n_locations': self.n_locations,
            'belief': self._belief.tolist() if self._belief is not None else None,
            'human_sighting_counts': self._human_sighting_counts.tolist(),
            'last_sighting_location': self._last_sighting_location,
            'frames_since_sighting': self._frames_since_sighting,
            'prior_alpha': self.prior_alpha,
        }

    @classmethod
    def load_state(cls, state: Dict) -> 'HumanSearchPOMDP':
        """Load from saved state."""
        pomdp = cls(
            n_locations=state['n_locations'],
            prior_alpha=state.get('prior_alpha', DIRICHLET_PRIOR_ALPHA),
        )

        if state['belief'] is not None:
            pomdp._belief = jnp.array(state['belief'])

        counts = state.get('human_sighting_counts', [])
        if counts:
            pomdp._human_sighting_counts = np.array(counts)

        pomdp._last_sighting_location = state.get('last_sighting_location')
        pomdp._frames_since_sighting = state.get('frames_since_sighting', 0)

        return pomdp
