"""
World Model POMDP for semantic localization.

Maintains a soft belief over learned topological locations, using:
- Observation likelihood from learned A matrix
- Transition dynamics from learned B matrix
- VFE for novelty detection (triggering new location discovery)
- EFE for exploration guidance

The world model answers: "Where am I?" with uncertainty quantification.
"""

import numpy as np
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from .config import (
    N_MAX_LOCATIONS, N_OBJECT_TYPES, N_OBS_LEVELS, N_MOVEMENT_ACTIONS,
    LOCATION_SIMILARITY_THRESHOLD, NOVELTY_SURPRISAL_THRESHOLD,
    VFE_HIGH_THRESHOLD, VFE_LOW_THRESHOLD, DIRICHLET_PRIOR_ALPHA
)
from .topological_map import TopologicalMap, LocationNode
from .observation_encoder import ObservationToken
from .similarity import cosine_similarity, batch_cosine_similarity
from .core import (
    normalize, belief_update, belief_update_with_vfe,
    surprisal, entropy, expected_free_energy, select_action
)


@dataclass
class LocalizationResult:
    """Result of a localization step."""
    # Current belief over locations
    belief: jnp.ndarray

    # Most likely location ID
    location_id: int

    # Confidence in current location (max belief)
    confidence: float

    # Whether a new location was discovered
    new_location_discovered: bool

    # VFE components for diagnostics
    vfe: float = 0.0
    accuracy: float = 0.0
    complexity: float = 0.0
    surprisal: float = 0.0

    # Similarity to best matching location
    similarity: float = 0.0


class WorldModel:
    """
    World Model POMDP for learned topological localization.

    Maintains:
    - Topological map of learned locations
    - Soft belief distribution over all locations
    - Online learning of observation and transition models

    The belief is updated each frame based on:
    1. Current observation (YOLO detections)
    2. Previous action taken (movement command)
    """

    def __init__(
        self,
        topo_map: TopologicalMap = None,
        novelty_threshold: float = NOVELTY_SURPRISAL_THRESHOLD,
        similarity_threshold: float = LOCATION_SIMILARITY_THRESHOLD
    ):
        """
        Initialize world model.

        Args:
            topo_map: Existing topological map (or None for fresh start)
            novelty_threshold: Surprisal threshold for new location detection
            similarity_threshold: Cosine similarity threshold for localization
        """
        self.topo_map = topo_map if topo_map is not None else TopologicalMap()
        self.novelty_threshold = novelty_threshold
        self.similarity_threshold = similarity_threshold

        # Current belief over locations (uniform if map exists, else empty)
        self._belief: Optional[jnp.ndarray] = None
        self._init_belief()

        # Previous location for transition learning
        self._prev_location_id: int = -1
        self._prev_action: int = -1

        # Cached matrices for JIT efficiency
        self._A_cache: Optional[jnp.ndarray] = None
        self._B_cache: Optional[jnp.ndarray] = None
        self._cache_valid: bool = False

    def _init_belief(self):
        """Initialize belief distribution."""
        n = self.topo_map.n_locations
        if n > 0:
            self._belief = jnp.ones(n) / n
        else:
            self._belief = None

    def _invalidate_cache(self):
        """Mark cached matrices as stale."""
        self._cache_valid = False

    def _update_cache(self):
        """Update cached A and B matrices if needed."""
        if not self._cache_valid:
            if self.topo_map.n_locations > 0:
                self._A_cache = self.topo_map.get_A_matrix()
                self._B_cache = self.topo_map.get_B_matrix()
            self._cache_valid = True

    @property
    def n_locations(self) -> int:
        """Number of known locations."""
        return self.topo_map.n_locations

    @property
    def belief(self) -> Optional[jnp.ndarray]:
        """Current belief distribution over locations."""
        return self._belief

    @property
    def current_location_id(self) -> int:
        """Most likely current location (-1 if unknown)."""
        if self._belief is None or len(self._belief) == 0:
            return -1
        return int(jnp.argmax(self._belief))

    @property
    def confidence(self) -> float:
        """Confidence in current location (max belief value)."""
        if self._belief is None or len(self._belief) == 0:
            return 0.0
        return float(jnp.max(self._belief))

    def localize(
        self,
        observation: ObservationToken,
        action_taken: int = 0
    ) -> LocalizationResult:
        """
        Update location belief based on new observation.

        This is the main update function called each frame.

        Args:
            observation: Current YOLO observation token
            action_taken: Movement action taken since last observation (0=stay)

        Returns:
            LocalizationResult with updated belief and diagnostics
        """
        obs_signature = observation.to_signature_vector()

        # Case 1: No locations yet - create first one
        if self.n_locations == 0:
            node = self.topo_map.add_node(observation)
            self._belief = jnp.ones(1)
            self._invalidate_cache()

            return LocalizationResult(
                belief=self._belief,
                location_id=0,
                confidence=1.0,
                new_location_discovered=True,
                similarity=1.0,
            )

        # Get observation index for A matrix lookup
        obs_idx = self._observation_to_index(observation)

        # Predict belief based on action (if we have transition model)
        if action_taken > 0 and self._prev_location_id >= 0:
            self._update_cache()
            if self._B_cache is not None and self._belief is not None:
                # B[:, :, action] gives transition probabilities
                B_action = self._B_cache[:, :, min(action_taken, N_MOVEMENT_ACTIONS - 1)]
                self._belief = jnp.dot(B_action, self._belief)
                self._belief = normalize(self._belief)

        # Check similarity to existing locations
        best_id, best_sim = self.topo_map.find_best_match(observation)

        # Compute VFE to detect novelty
        self._update_cache()
        vfe, acc, comp, surp = 0.0, 0.0, 0.0, 0.0

        if self._A_cache is not None and self._belief is not None:
            # Belief update with VFE monitoring
            likelihood = self._get_observation_likelihood(observation)
            new_belief, vfe, acc, comp, surp = belief_update_with_vfe(
                self._belief,
                self._A_cache,
                obs_idx
            )
        else:
            new_belief = self._belief
            likelihood = None

        # Decision: new location or update existing?
        new_location = False

        if best_sim < self.similarity_threshold or surp > self.novelty_threshold:
            # Novel observation - add new location
            if self.n_locations < N_MAX_LOCATIONS:
                node = self.topo_map.add_node(observation)
                new_location = True

                # Expand belief to include new location
                if self._belief is not None:
                    # New location gets some probability mass
                    new_prob = 0.3  # Initial probability for new location
                    old_belief = self._belief * (1 - new_prob)
                    self._belief = jnp.concatenate([old_belief, jnp.array([new_prob])])
                else:
                    self._belief = jnp.ones(1)

                self._invalidate_cache()
                best_id = node.id
                best_sim = 1.0
        else:
            # Update existing location
            node = self.topo_map.get_node(best_id)
            if node is not None:
                node.update_signature(obs_signature)
                node.update_A_counts(observation.object_levels)
                self._invalidate_cache()

            # Update belief with observation
            if likelihood is not None and new_belief is not None:
                self._belief = new_belief

        # Learn transition if we moved
        if (self._prev_location_id >= 0 and
            self._prev_location_id != best_id and
            self._prev_action > 0):
            self.topo_map.record_transition(
                self._prev_location_id,
                best_id,
                self._prev_action
            )
            self._invalidate_cache()

        # Store for next iteration
        self._prev_location_id = best_id
        self._prev_action = action_taken

        # Update map's current location
        self.topo_map.current_location_id = best_id

        return LocalizationResult(
            belief=self._belief,
            location_id=best_id,
            confidence=float(jnp.max(self._belief)) if self._belief is not None else 0.0,
            new_location_discovered=new_location,
            vfe=float(vfe),
            accuracy=float(acc),
            complexity=float(comp),
            surprisal=float(surp),
            similarity=best_sim,
        )

    def _observation_to_index(self, observation: ObservationToken) -> int:
        """
        Convert observation to flat index for A matrix lookup.

        Maps (object_type, obs_level) to single index.
        """
        # Find the most distinctive observed object
        max_level = 0
        max_type = 0
        for i, level in enumerate(observation.object_levels):
            if level > max_level:
                max_level = level
                max_type = i

        # Flat index: type * N_OBS_LEVELS + level
        return max_type * N_OBS_LEVELS + max_level

    def _get_observation_likelihood(
        self,
        observation: ObservationToken
    ) -> jnp.ndarray:
        """
        Get observation likelihood for each location.

        Uses cosine similarity as likelihood proxy when A matrix is sparse.
        """
        if self.n_locations == 0:
            return jnp.array([])

        obs_sig = jnp.array(observation.to_signature_vector())
        all_sigs = jnp.array(self.topo_map.get_all_signatures())

        # Compute similarities as likelihoods
        likelihoods = batch_cosine_similarity(obs_sig, all_sigs)

        # Ensure non-negative and normalized
        likelihoods = jnp.maximum(likelihoods, 0.01)
        return likelihoods

    def get_exploration_target(
        self,
        preferences: jnp.ndarray = None
    ) -> Tuple[int, jnp.ndarray]:
        """
        Get recommended action for exploration using EFE.

        Args:
            preferences: Optional observation preferences (C matrix)

        Returns:
            (best_action, action_probabilities)
        """
        if self.n_locations == 0 or self._belief is None:
            return 0, jnp.ones(N_MOVEMENT_ACTIONS) / N_MOVEMENT_ACTIONS

        self._update_cache()

        if self._A_cache is None or self._B_cache is None:
            return 0, jnp.ones(N_MOVEMENT_ACTIONS) / N_MOVEMENT_ACTIONS

        # Default preferences: prefer high-entropy (informative) observations
        if preferences is None:
            n_obs = self._A_cache.shape[0]
            preferences = jnp.ones(n_obs) / n_obs

        return select_action(
            self._belief,
            self._A_cache,
            self._B_cache,
            preferences,
            N_MOVEMENT_ACTIONS
        )

    def get_location_info(self, location_id: int = None) -> Dict[str, Any]:
        """
        Get information about a location.

        Args:
            location_id: Location to query (default: current)

        Returns:
            Dict with location details
        """
        if location_id is None:
            location_id = self.current_location_id

        if location_id < 0 or location_id >= self.n_locations:
            return {'error': 'Invalid location'}

        node = self.topo_map.get_node(location_id)
        if node is None:
            return {'error': 'Location not found'}

        # Find top objects at this location
        from .config import TYPE_NAMES
        sig = node.observation_signature
        top_objects = []
        sorted_idx = sig.argsort()[::-1]
        for idx in sorted_idx[:5]:
            if sig[idx] > 0.1:
                top_objects.append({
                    'name': TYPE_NAMES[idx],
                    'score': float(sig[idx])
                })

        neighbors = self.topo_map.get_neighbors(location_id)

        return {
            'id': location_id,
            'visit_count': node.visit_count,
            'established': node.is_established(),
            'top_objects': top_objects,
            'neighbors': [
                {'to': n[0], 'action': n[1], 'count': n[2]}
                for n in neighbors
            ],
            'belief': float(self._belief[location_id]) if self._belief is not None else 0.0,
        }

    def get_belief_entropy(self) -> float:
        """
        Get entropy of location belief (uncertainty measure).

        High entropy = uncertain about location
        Low entropy = confident about location
        """
        if self._belief is None:
            return 0.0
        return float(entropy(self._belief))

    def reset_belief(self):
        """Reset belief to uniform over known locations."""
        self._init_belief()
        self._prev_location_id = -1
        self._prev_action = -1

    def save(self, name: str = None) -> str:
        """Save the learned map."""
        from .map_persistence import save_map
        path = save_map(self.topo_map, name=name)
        return str(path)

    @classmethod
    def load(cls, name: str = None, filepath: str = None) -> 'WorldModel':
        """Load a world model from saved map."""
        from .map_persistence import load_map, load_latest_map

        if filepath:
            from pathlib import Path
            topo_map = load_map(filepath=Path(filepath))
        elif name:
            topo_map = load_map(name=name)
        else:
            topo_map = load_latest_map()

        if topo_map is None:
            return cls()

        return cls(topo_map=topo_map)

    def __repr__(self) -> str:
        return (
            f"WorldModel(n_locations={self.n_locations}, "
            f"current={self.current_location_id}, "
            f"confidence={self.confidence:.2f})"
        )
