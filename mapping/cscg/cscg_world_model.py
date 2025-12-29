"""
CSCG-based World Model for POMDP integration.

Wraps CHMM with a POMDP-compatible interface, providing:
- Localization via clone state inference
- VFE computation from belief dynamics
- Integration with observation tokenizer

Supports multiple tokenization backends:
- CLIP: Semantic embeddings (512 dim) - global image features
- DINOv2: Structural features (384 dim) - better than CLIP
- ORB: Local keypoint features - BEST for place recognition
"""

import numpy as np
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

from .chmm import CHMM
from .tokenizer import EmbeddingTokenizer, HybridTokenizer
from ..vbgs_place import PlaceManager


@dataclass
class CSCGLocalizationResult:
    """Result of CSCG localization step."""
    # Belief over clone states
    belief: np.ndarray

    # Most likely clone state
    clone_state: int

    # Observation token
    token: int

    # Token similarity
    token_similarity: float

    # Confidence (max belief)
    confidence: float

    # Whether a new token was created
    new_token_discovered: bool

    # VFE components
    vfe: float = 0.0
    accuracy: float = 0.0
    complexity: float = 0.0
    surprisal: float = 0.0

    # Compatibility aliases for existing code
    @property
    def location_id(self) -> int:
        """Alias for clone_state (compatibility with standard LocalizationResult)."""
        return self.clone_state

    @property
    def similarity(self) -> float:
        """Alias for token_similarity (compatibility with standard LocalizationResult)."""
        return self.token_similarity

    @property
    def new_location_discovered(self) -> bool:
        """Alias for new_token_discovered (compatibility with standard LocalizationResult)."""
        return self.new_token_discovered


class CSCGWorldModel:
    """
    World Model using Clone-Structured Cognitive Graphs.

    Provides POMDP-compatible interface for:
    - Localization: "Where am I?" via clone state inference
    - Learning: Online EM updates for transition structure
    - VFE: Variational Free Energy for novelty detection
    - Path planning: Bridge function for navigation

    Attributes:
        chmm: Clone-structured Hidden Markov Model
        tokenizer: Embedding to discrete token converter
        belief: Current belief over clone states
    """

    def __init__(
        self,
        n_tokens: int = 32,
        n_clones_per_token: int = 3,
        n_actions: int = 5,
        embedding_dim: int = 384,  # DINOv2-small default
        similarity_threshold: float = 0.85,
        use_hybrid_tokenizer: bool = False,  # Disabled by default for ORB
        n_object_types: int = 17,
        expected_places: int = 5,  # Prior on how many places we expect to find
        encoder_type: str = "orb",  # "clip", "dinov2", or "orb" (RECOMMENDED)
    ):
        """
        Initialize CSCG World Model.

        Args:
            n_tokens: Maximum observation tokens
            n_clones_per_token: Clone states per token (for aliasing disambiguation)
            n_actions: Number of action types (e.g., stay/fwd/back/left/right)
            embedding_dim: Embedding dimension (512 for CLIP, 384 for DINOv2-small)
            similarity_threshold: Token assignment threshold
            use_hybrid_tokenizer: Use encoder + YOLO hybrid tokens (ignored for ORB)
            n_object_types: Number of YOLO object types (for hybrid tokenizer)
            expected_places: Prior belief about how many places exist (for exploration)
            encoder_type: Which encoder to use ("clip", "dinov2", or "orb")
                          ORB is RECOMMENDED - 87.5% accuracy vs 50% for semantic encoders
        """
        self.expected_places = expected_places
        self.n_tokens = n_tokens
        self.n_clones_per_token = n_clones_per_token
        self.n_actions = n_actions
        self.encoder_type = encoder_type

        # ORB doesn't use embedding tokenizers - it directly produces place IDs
        self.use_orb = (encoder_type == "orb")
        self.use_hybrid_tokenizer = use_hybrid_tokenizer and not self.use_orb

        if self.use_orb:
            # ORB place recognizer - directly tokenizes to place IDs
            from pomdp.place_recognizer import ORBPlaceRecognizer
            self._orb_recognizer = ORBPlaceRecognizer(
                n_features=500,
                match_threshold=0.75,
                min_matches=15,
                keyframe_cooldown=10,
                max_keyframes=n_tokens,
            )
            self.tokenizer = None  # Not used for ORB
        else:
            # Embedding-based tokenizer (CLIP/DINOv2)
            self._orb_recognizer = None

            # Set embedding dim based on encoder if not specified
            if encoder_type == "dinov2" and embedding_dim == 512:
                embedding_dim = 384  # DINOv2-small default

            # Tokenizer
            if self.use_hybrid_tokenizer:
                self.tokenizer = HybridTokenizer(
                    n_tokens=n_tokens,
                    embedding_dim=embedding_dim,
                    n_object_types=n_object_types,
                    similarity_threshold=similarity_threshold,
                )
            else:
                self.tokenizer = EmbeddingTokenizer(
                    n_tokens=n_tokens,
                    embedding_dim=embedding_dim,
                    similarity_threshold=similarity_threshold,
                )

        # CHMM will be initialized when first token is created
        self._chmm: Optional[CHMM] = None

        # Current belief over clone states
        self._belief: Optional[np.ndarray] = None

        # Previous state for transition tracking
        self._prev_token: int = -1
        self._prev_action: int = 0

        # Sequence buffer for batch learning
        self._token_history: List[int] = []
        self._action_history: List[int] = []
        self._max_history = 1000

        # Obstacle tracking: blocked_actions[place_id][action] = count
        # This helps the agent avoid recommending actions that don't work
        self._blocked_actions: Dict[int, Dict[int, int]] = {}

        # Image encoder (lazy loaded) - only for CLIP/DINOv2
        self._image_encoder = None

        # Embedding dim for place manager
        if self.use_orb:
            embedding_dim = 32  # ORB descriptors are 32 bytes

        # Place manager for VBGS local place models
        self.place_manager = PlaceManager(
            use_simple_model=True,
            embedding_dim=embedding_dim,
        )

    def _get_image_encoder(self):
        """Lazy load image encoder based on encoder_type."""
        if self._image_encoder is None:
            if self.encoder_type == "dinov2":
                from pomdp.image_encoder import DINOv2Encoder
                self._image_encoder = DINOv2Encoder()
            else:  # "clip" or default
                from pomdp.image_encoder import ImageEncoder
                self._image_encoder = ImageEncoder()
        return self._image_encoder

    def _ensure_chmm(self, n_tokens: int):
        """Initialize or expand CHMM to accommodate tokens."""
        if self._chmm is None:
            # Initialize with uniform clones per token
            n_clones = np.ones(n_tokens, dtype=np.int32) * self.n_clones_per_token
            self._chmm = CHMM(
                n_clones=n_clones,
                n_actions=self.n_actions,
            )
            self._belief = np.ones(self._chmm.n_states) / self._chmm.n_states

        elif n_tokens > len(self._chmm.n_clones):
            # Expand CHMM for new tokens
            old_n_clones = self._chmm.n_clones.copy()
            old_T = self._chmm.T.copy()

            # New clone counts
            new_n_clones = np.ones(n_tokens, dtype=np.int32) * self.n_clones_per_token
            new_n_clones[:len(old_n_clones)] = old_n_clones

            # Create expanded CHMM
            self._chmm = CHMM(
                n_clones=new_n_clones,
                n_actions=self.n_actions,
            )

            # Copy old transitions
            old_n_states = old_T.shape[0]
            self._chmm.T[:old_n_states, :old_n_states, :] = old_T

            # Expand belief
            old_belief = self._belief
            self._belief = np.ones(self._chmm.n_states) / self._chmm.n_states
            if old_belief is not None:
                self._belief[:len(old_belief)] = old_belief
                self._belief /= self._belief.sum()

    def localize(
        self,
        frame: np.ndarray,
        action_taken: int = 0,
        observation_token: Optional['ObservationToken'] = None,
        debug: bool = False,
    ) -> CSCGLocalizationResult:
        """
        Update location belief based on new observation.

        Args:
            frame: BGR image frame
            action_taken: Movement action taken (0=stay, 1=fwd, 2=back, 3=left, 4=right)
            observation_token: Optional YOLO observation token (for hybrid tokenizer)
            debug: Print debug info

        Returns:
            CSCGLocalizationResult with updated belief and diagnostics
        """
        self._frame_count = getattr(self, '_frame_count', 0) + 1

        # Tokenization depends on encoder type
        if self.use_orb:
            # ORB-based tokenization (RECOMMENDED)
            place_id, place_name, confidence, is_new = self._orb_recognizer.recognize(
                frame, allow_new=True, debug=debug
            )
            token = place_id
            token_sim = confidence
            new_token = is_new
            n_active = self._orb_recognizer.n_places
            embedding = None  # ORB doesn't produce embeddings

            if debug or (self._frame_count % 60 == 0):
                print(f"  [CSCG-ORB] place={place_name}(id={token}), conf={token_sim:.3f}, n_places={n_active}, new={new_token}")
        else:
            # Embedding-based tokenization (CLIP/DINOv2)
            encoder = self._get_image_encoder()
            embedding = encoder.encode(frame, temporal_smooth=True)

            if self.use_hybrid_tokenizer and observation_token is not None:
                obj_hist = observation_token.to_signature_vector()
                token, token_sim = self.tokenizer.tokenize(embedding, obj_hist)
            else:
                token, token_sim = self.tokenizer.tokenize(embedding)

            new_token = (token >= self.tokenizer.n_active - 1 and
                         self.tokenizer.counts[token] <= 1)
            n_active = self.tokenizer.n_active

            if debug or (self._frame_count % 60 == 0):
                print(f"  [CSCG] token={token}, sim={token_sim:.3f}, n_active={n_active}, new={new_token}")

        # 3. Ensure CHMM is sized for current tokens
        self._ensure_chmm(n_active)

        # 4. Belief update
        prev_belief = self._belief.copy() if self._belief is not None else None
        self._belief = self._chmm.belief(token, prev_belief, action_taken)

        # 5. Compute VFE
        vfe, acc, comp, surp = self._compute_vfe(self._belief, prev_belief, token)

        # 6. Record for learning
        self._token_history.append(token)
        self._action_history.append(action_taken)
        if len(self._token_history) > self._max_history:
            self._token_history.pop(0)
            self._action_history.pop(0)

        # 7. Store for next step
        self._prev_token = token
        self._prev_action = action_taken

        # 8. Find best clone state
        clone_state = int(np.argmax(self._belief))
        confidence = float(self._belief[clone_state])

        # 9. Update place manager with embedding (only for embedding-based encoders)
        if embedding is not None:
            self.place_manager.update_place(
                place_id=token,
                embedding=embedding,
            )

        return CSCGLocalizationResult(
            belief=self._belief.copy(),
            clone_state=clone_state,
            token=token,
            token_similarity=token_sim,
            confidence=confidence,
            new_token_discovered=new_token,
            vfe=vfe,
            accuracy=acc,
            complexity=comp,
            surprisal=surp,
        )

    def _compute_vfe(
        self,
        posterior: np.ndarray,
        prior: Optional[np.ndarray],
        token: int,
    ) -> Tuple[float, float, float, float]:
        """
        Compute Variational Free Energy.

        VFE = Complexity - Accuracy
        - Complexity: KL[posterior || prior]
        - Accuracy: log p(token | clone_state) averaged over belief

        Returns:
            (vfe, accuracy, complexity, surprisal)
        """
        if prior is None:
            return 0.0, 0.0, 0.0, 0.0

        eps = 1e-10

        # Complexity: KL divergence
        kl = np.sum(posterior * (np.log(posterior + eps) - np.log(prior + eps)))
        complexity = float(kl)

        # Accuracy: expected log-likelihood of observation
        # For CSCG: only states that emit this token have P(token|state) = 1
        valid_states = self._chmm.obs_to_states[token]
        log_lik = np.log(len(valid_states) / self._chmm.n_states + eps)
        accuracy = float(np.sum(posterior[valid_states]) * log_lik)

        # VFE
        vfe = complexity - accuracy

        # Surprisal: -log P(token | prior)
        prior_prob = np.sum(prior[valid_states])
        surprisal = -np.log(prior_prob + eps)

        return vfe, accuracy, complexity, float(surprisal)

    def learn(self, n_iter: int = 5):
        """
        Run EM learning on accumulated history.

        Call periodically to update transition structure.
        """
        if self._chmm is None or len(self._token_history) < 10:
            return

        x = np.array(self._token_history, dtype=np.int32)
        a = np.array(self._action_history, dtype=np.int32)

        self._chmm.learn_em_T(x, a, n_iter=n_iter)

    def get_exploration_urgency(self) -> Tuple[float, str]:
        """
        Get exploration urgency based on model uncertainty.

        Returns:
            (urgency, reason) where urgency is 0-1 (1 = definitely explore more)
        """
        n_found = self.n_locations
        n_expected = self.expected_places
        history_len = len(self._token_history)

        # Factor 1: Haven't found all expected places
        place_ratio = n_found / max(n_expected, 1)
        place_urgency = max(0, 1.0 - place_ratio)

        # Factor 2: Not enough data collected yet
        # Need significant exploration history before considering "done"
        min_history = 500  # At least 500 observations
        data_urgency = max(0, 1.0 - history_len / min_history)

        # Factor 3: Belief entropy (high entropy = uncertain = explore)
        if self._belief is not None and len(self._belief) > 1:
            # Normalized entropy
            entropy = -np.sum(self._belief * np.log(self._belief + 1e-10))
            max_entropy = np.log(len(self._belief))
            belief_urgency = entropy / max_entropy if max_entropy > 0 else 0
        else:
            belief_urgency = 1.0  # No belief yet, definitely explore

        # Factor 4: Transition uncertainty (unexplored connections)
        if self._chmm is not None:
            # Count how many transitions are still uniform (unexplored)
            T = self._chmm.T
            n_states = T.shape[0]
            uniform_threshold = 0.8 / n_states  # Close to uniform

            unexplored_count = 0
            total_transitions = 0
            for a in range(self.n_actions):
                for s in range(n_states):
                    row = T[:, s, a]
                    if row.sum() > 0:
                        row_norm = row / row.sum()
                        # Check if close to uniform
                        if np.max(row_norm) < uniform_threshold * 2:
                            unexplored_count += 1
                        total_transitions += 1

            transition_urgency = unexplored_count / max(total_transitions, 1)
        else:
            transition_urgency = 1.0

        # Combine factors - data collection is most important early on
        urgency = 0.3 * place_urgency + 0.35 * data_urgency + 0.15 * belief_urgency + 0.2 * transition_urgency

        # Generate reason
        if data_urgency > 0.5:
            reason = f"Need more data ({history_len}/{min_history} observations)"
        elif place_urgency > 0.5:
            reason = f"Only found {n_found}/{n_expected} expected places"
        elif belief_urgency > 0.5:
            reason = "High uncertainty about current location"
        elif transition_urgency > 0.5:
            reason = "Many unexplored transitions"
        else:
            reason = "World model is well-explored"

        return float(urgency), reason

    def should_explore(self) -> bool:
        """Check if we should continue exploring based on model uncertainty."""
        urgency, _ = self.get_exploration_urgency()
        return urgency > 0.3  # Threshold for "should explore"

    def get_exploration_target(self) -> Tuple[int, np.ndarray]:
        """
        Get recommended action for exploration.

        Uses entropy reduction heuristic: prefer actions that reduce uncertainty.
        Also penalizes actions that have been blocked at the current place.

        Returns:
            (best_action, action_scores)
        """
        if self._chmm is None or self._belief is None:
            return 0, np.ones(self.n_actions) / self.n_actions

        action_scores = np.zeros(self.n_actions)

        for action in range(self.n_actions):
            # Predict belief after action
            predicted = self._chmm.T[:, :, action] @ self._belief

            # Normalize
            norm = predicted.sum()
            if norm > 0:
                predicted /= norm

            # Score: negative entropy (lower = more certain = better)
            entropy = -np.sum(predicted * np.log(predicted + 1e-10))
            action_scores[action] = -entropy

        # Apply obstacle penalty: actions that have been blocked get lower scores
        block_penalties = self.get_blocked_penalty()
        # Each block reduces the score significantly
        action_scores = action_scores - block_penalties * 2.0

        # Softmax for probabilities
        action_scores = action_scores - action_scores.max()
        action_probs = np.exp(action_scores)
        action_probs /= action_probs.sum()

        best_action = int(np.argmax(action_probs))
        return best_action, action_probs

    def record_blocked_action(self, action: int, place_id: int = None):
        """
        Record that an action was blocked (hit obstacle) at a place.

        This information is used to penalize actions that don't work,
        so the agent learns to avoid walls/obstacles.

        Args:
            action: The action that was blocked (1=forward, 2=backward, etc.)
            place_id: The place where blocking occurred (defaults to current place)
        """
        if place_id is None:
            place_id = self._prev_token

        if place_id < 0:
            return  # No valid place yet

        if place_id not in self._blocked_actions:
            self._blocked_actions[place_id] = {}

        if action not in self._blocked_actions[place_id]:
            self._blocked_actions[place_id][action] = 0

        self._blocked_actions[place_id][action] += 1

    def get_blocked_penalty(self, place_id: int = None) -> np.ndarray:
        """
        Get penalty scores for blocked actions at a place.

        Returns array of penalties (higher = more blocked = avoid).

        Args:
            place_id: Place to check (defaults to current place)

        Returns:
            Array of shape (n_actions,) with block counts
        """
        if place_id is None:
            place_id = self._prev_token

        penalties = np.zeros(self.n_actions)

        if place_id in self._blocked_actions:
            for action, count in self._blocked_actions[place_id].items():
                if action < self.n_actions:
                    penalties[action] = count

        return penalties

    def clear_blocked_actions(self, place_id: int = None):
        """Clear blocked action history for a place (or all places)."""
        if place_id is None:
            self._blocked_actions.clear()
        elif place_id in self._blocked_actions:
            del self._blocked_actions[place_id]

    def bridge(self, target_token: int, max_steps: int = 50) -> Optional[List[int]]:
        """
        Find action sequence to reach a target observation token.

        Args:
            target_token: Target observation token
            max_steps: Maximum path length

        Returns:
            List of actions, or None if no path found
        """
        if self._chmm is None:
            return None

        # Current state
        current_state = int(np.argmax(self._belief))

        # Target states (any clone of target token)
        if target_token >= len(self._chmm.obs_to_states):
            return None

        target_states = self._chmm.obs_to_states[target_token]

        # Try to find path to any target state
        for target_state in target_states:
            path = self._chmm.bridge(current_state, target_state, max_steps)
            if path is not None:
                return [action for _, action in path]

        return None

    def get_clone_belief(self) -> np.ndarray:
        """Get current belief over clone states."""
        if self._belief is None:
            return np.array([])
        return self._belief.copy()

    def get_token_belief(self) -> np.ndarray:
        """Get belief aggregated by observation token (marginalizing over clones)."""
        if self._chmm is None or self._belief is None:
            return np.array([])

        n_tokens = self.n_locations
        token_belief = np.zeros(n_tokens)
        for token in range(n_tokens):
            if token < len(self._chmm.obs_to_states):
                states = self._chmm.obs_to_states[token]
                token_belief[token] = np.sum(self._belief[states])

        return token_belief

    @property
    def n_locations(self) -> int:
        """Number of active tokens (analogous to locations)."""
        if self.use_orb:
            return self._orb_recognizer.n_places
        return self.tokenizer.n_active

    @property
    def n_clone_states(self) -> int:
        """Total number of clone states."""
        if self._chmm is None:
            return 0
        return self._chmm.n_states

    @property
    def current_location_id(self) -> int:
        """Most likely clone state."""
        if self._belief is None:
            return -1
        return int(np.argmax(self._belief))

    @property
    def current_clone_state(self) -> int:
        """Most likely clone state (alias for current_location_id)."""
        return self.current_location_id

    @property
    def current_token(self) -> int:
        """Current observation token."""
        return self._prev_token

    @property
    def confidence(self) -> float:
        """Confidence in current location."""
        if self._belief is None:
            return 0.0
        return float(np.max(self._belief))

    def get_location_info(self, clone_state: int = None) -> Dict[str, Any]:
        """Get information about a clone state."""
        if clone_state is None:
            clone_state = self.current_location_id

        if self._chmm is None or clone_state < 0:
            return {'error': 'Invalid state'}

        token = self._chmm.state_to_obs[clone_state]

        # Get token count based on encoder type
        if self.use_orb:
            # Get visit count from ORB keyframe
            token_count = 0
            place_name = "Unknown"
            for kf in self._orb_recognizer.keyframes:
                if kf.id == token:
                    token_count = kf.visit_count
                    place_name = kf.name
                    break
        else:
            token_count = int(self.tokenizer.counts[token])
            place_name = f"Token_{token}"

        return {
            'clone_state': clone_state,
            'token': int(token),
            'token_count': token_count,
            'place_name': place_name,
            'belief': float(self._belief[clone_state]) if self._belief is not None else 0.0,
            'n_clones_for_token': len(self._chmm.obs_to_states[token]),
        }

    def get_place_name(self, token: int = None) -> str:
        """Get human-readable name for a place/token."""
        if token is None:
            token = self._prev_token

        if self.use_orb:
            return self._orb_recognizer.get_place_name(token)
        return f"Token_{token}"

    def reset_belief(self):
        """Reset belief to uniform."""
        if self._chmm is not None:
            self._belief = np.ones(self._chmm.n_states) / self._chmm.n_states
        self._prev_token = -1
        self._prev_action = 0

    def save(self, directory: str):
        """Save model to directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        self.tokenizer.save(str(path / 'tokenizer.npz'))

        # Save CHMM
        if self._chmm is not None:
            self._chmm.save(str(path / 'chmm.npz'))

        # Save history
        np.savez(
            str(path / 'history.npz'),
            tokens=np.array(self._token_history, dtype=np.int32),
            actions=np.array(self._action_history, dtype=np.int32),
        )

        # Save config
        config = {
            'n_tokens': self.n_tokens,
            'n_clones_per_token': self.n_clones_per_token,
            'n_actions': self.n_actions,
            'use_hybrid_tokenizer': self.use_hybrid_tokenizer,
        }
        np.savez(str(path / 'config.npz'), **config)

    @classmethod
    def load(cls, directory: str) -> 'CSCGWorldModel':
        """Load model from directory."""
        path = Path(directory)

        # Load config
        config = dict(np.load(str(path / 'config.npz')))

        # Create model
        model = cls(
            n_tokens=int(config['n_tokens']),
            n_clones_per_token=int(config['n_clones_per_token']),
            n_actions=int(config['n_actions']),
            use_hybrid_tokenizer=bool(config['use_hybrid_tokenizer']),
        )

        # Load tokenizer
        if model.use_hybrid_tokenizer:
            model.tokenizer = HybridTokenizer.load(str(path / 'tokenizer.npz'))
        else:
            model.tokenizer = EmbeddingTokenizer.load(str(path / 'tokenizer.npz'))

        # Load CHMM
        if (path / 'chmm.npz').exists():
            model._chmm = CHMM.load(str(path / 'chmm.npz'))
            model._belief = np.ones(model._chmm.n_states) / model._chmm.n_states

        # Load history
        if (path / 'history.npz').exists():
            history = np.load(str(path / 'history.npz'))
            model._token_history = history['tokens'].tolist()
            model._action_history = history['actions'].tolist()

        return model

    def __repr__(self) -> str:
        return (f"CSCGWorldModel(n_tokens={self.n_locations}, "
                f"n_clones={self.n_clone_states}, "
                f"confidence={self.confidence:.2f})")
