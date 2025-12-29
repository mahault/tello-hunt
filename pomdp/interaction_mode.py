"""
Interaction Mode POMDP for action selection via Expected Free Energy.

This module maintains:
- Belief over engagement states (searching, approaching, interacting, disengaging)
- Action selection via EFE minimization
- Transition model for engagement dynamics

State space: 4 engagement states
Observation: Combined person detection + proximity info
Action space: 6 drone actions
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from .config import (
    ENGAGEMENT_STATES,
    N_ENGAGEMENT_STATES,
    ACTIONS,
    N_ACTIONS,
    N_PERSON_OBS,
    PERSON_OBS,
    ACTION_TEMPERATURE,
    DIRICHLET_PRIOR_ALPHA,
)
from .core import (
    normalize,
    softmax,
    entropy,
    belief_update,
    predict_next_belief,
    predict_observation,
    expected_free_energy,
    compute_all_efe,
    select_action,
)
from .observation_encoder import ObservationToken


# Engagement state indices
SEARCHING = 0
APPROACHING = 1
INTERACTING = 2
DISENGAGING = 3

# Action indices
ACTION_CONTINUE_SEARCH = 0
ACTION_APPROACH = 1
ACTION_INTERACT_LED = 2
ACTION_INTERACT_WIGGLE = 3
ACTION_BACKOFF = 4
ACTION_LAND = 5


@dataclass
class InteractionResult:
    """Result of interaction mode update and action selection."""
    belief: jnp.ndarray  # Belief over engagement states (4,)
    engagement_state: str  # Most likely state name
    confidence: float  # Max probability
    selected_action: str  # Selected action name
    selected_action_idx: int  # Selected action index
    action_probs: jnp.ndarray  # Probability distribution over actions
    efe_values: jnp.ndarray  # EFE for each action (lower = better)
    belief_entropy: float  # Uncertainty over engagement state


class InteractionModePOMDP:
    """
    POMDP for selecting drone actions via active inference.

    State: Engagement mode (searching, approaching, interacting, disengaging)
    Observation: Person detection category (from YOLO)
    Action: Drone control action (continue_search, approach, interact, backoff, land)

    Uses Expected Free Energy (EFE) to balance:
    - Epistemic value: reducing uncertainty about engagement state
    - Pragmatic value: achieving preferred outcomes (finding/interacting with humans)
    """

    def __init__(
        self,
        prior_alpha: float = DIRICHLET_PRIOR_ALPHA,
        temperature: float = ACTION_TEMPERATURE,
    ):
        """
        Initialize interaction mode POMDP.

        Args:
            prior_alpha: Dirichlet prior for learning
            temperature: Action selection temperature (lower = more deterministic)
        """
        self.prior_alpha = prior_alpha
        self.temperature = temperature

        # Belief over engagement states
        # Start in "searching" mode
        self._belief: jnp.ndarray = jnp.array([0.7, 0.1, 0.1, 0.1])

        # Build fixed matrices (could be learned, but start with hand-designed)
        self._A = self._build_A_matrix()
        self._B = self._build_B_matrix()
        self._C = self._build_C_preferences()

        # Track last action for transition updates
        self._last_action: Optional[int] = None

        # Statistics
        self._frames_in_state: Dict[int, int] = {i: 0 for i in range(N_ENGAGEMENT_STATES)}
        self._total_actions: Dict[int, int] = {i: 0 for i in range(N_ACTIONS)}

    @property
    def belief(self) -> jnp.ndarray:
        """Current belief over engagement states."""
        return self._belief

    def _build_A_matrix(self) -> jnp.ndarray:
        """
        Build observation likelihood matrix P(person_obs | engagement_state).

        Encodes what observations we expect in each engagement state.

        Returns:
            A: Shape (N_PERSON_OBS, N_ENGAGEMENT_STATES)
        """
        # Rows: observations (not_detected, left, center, right, close)
        # Cols: states (searching, approaching, interacting, disengaging)
        A = np.zeros((N_PERSON_OBS, N_ENGAGEMENT_STATES))

        # Observation indices
        NOT_DETECTED = 0
        DETECTED_LEFT = 1
        DETECTED_CENTER = 2
        DETECTED_RIGHT = 3
        DETECTED_CLOSE = 4

        # SEARCHING: Usually no person detected (we're looking)
        A[NOT_DETECTED, SEARCHING] = 0.70
        A[DETECTED_LEFT, SEARCHING] = 0.08
        A[DETECTED_CENTER, SEARCHING] = 0.10
        A[DETECTED_RIGHT, SEARCHING] = 0.08
        A[DETECTED_CLOSE, SEARCHING] = 0.04

        # APPROACHING: Person detected, usually center or sides (tracking them)
        A[NOT_DETECTED, APPROACHING] = 0.10
        A[DETECTED_LEFT, APPROACHING] = 0.20
        A[DETECTED_CENTER, APPROACHING] = 0.40
        A[DETECTED_RIGHT, APPROACHING] = 0.20
        A[DETECTED_CLOSE, APPROACHING] = 0.10

        # INTERACTING: Person close (we've reached them)
        A[NOT_DETECTED, INTERACTING] = 0.05
        A[DETECTED_LEFT, INTERACTING] = 0.05
        A[DETECTED_CENTER, INTERACTING] = 0.20
        A[DETECTED_RIGHT, INTERACTING] = 0.05
        A[DETECTED_CLOSE, INTERACTING] = 0.65

        # DISENGAGING: Person moving away or lost
        A[NOT_DETECTED, DISENGAGING] = 0.40
        A[DETECTED_LEFT, DISENGAGING] = 0.15
        A[DETECTED_CENTER, DISENGAGING] = 0.20
        A[DETECTED_RIGHT, DISENGAGING] = 0.15
        A[DETECTED_CLOSE, DISENGAGING] = 0.10

        return jnp.array(A)

    def _build_B_matrix(self) -> jnp.ndarray:
        """
        Build transition matrix P(state' | state, action).

        Encodes how actions affect engagement state transitions.

        Returns:
            B: Shape (N_ENGAGEMENT_STATES, N_ENGAGEMENT_STATES, N_ACTIONS)
        """
        B = np.zeros((N_ENGAGEMENT_STATES, N_ENGAGEMENT_STATES, N_ACTIONS))

        # Default: mostly stay in current state with small drift
        for a in range(N_ACTIONS):
            B[:, :, a] = np.eye(N_ENGAGEMENT_STATES) * 0.6
            B[:, :, a] += 0.1  # Small probability of any transition

        # ACTION: continue_search
        # Searching -> stays searching, might find person -> approaching
        B[SEARCHING, SEARCHING, ACTION_CONTINUE_SEARCH] = 0.70
        B[APPROACHING, SEARCHING, ACTION_CONTINUE_SEARCH] = 0.25
        B[INTERACTING, SEARCHING, ACTION_CONTINUE_SEARCH] = 0.02
        B[DISENGAGING, SEARCHING, ACTION_CONTINUE_SEARCH] = 0.03
        # If approaching and keep searching, might lose target
        B[SEARCHING, APPROACHING, ACTION_CONTINUE_SEARCH] = 0.30
        B[APPROACHING, APPROACHING, ACTION_CONTINUE_SEARCH] = 0.50
        B[INTERACTING, APPROACHING, ACTION_CONTINUE_SEARCH] = 0.05
        B[DISENGAGING, APPROACHING, ACTION_CONTINUE_SEARCH] = 0.15

        # ACTION: approach
        # Should transition toward interacting
        B[SEARCHING, SEARCHING, ACTION_APPROACH] = 0.20
        B[APPROACHING, SEARCHING, ACTION_APPROACH] = 0.70
        B[INTERACTING, SEARCHING, ACTION_APPROACH] = 0.05
        B[DISENGAGING, SEARCHING, ACTION_APPROACH] = 0.05
        # Approaching -> might reach interaction
        B[SEARCHING, APPROACHING, ACTION_APPROACH] = 0.05
        B[APPROACHING, APPROACHING, ACTION_APPROACH] = 0.50
        B[INTERACTING, APPROACHING, ACTION_APPROACH] = 0.40
        B[DISENGAGING, APPROACHING, ACTION_APPROACH] = 0.05
        # If interacting and approach more
        B[SEARCHING, INTERACTING, ACTION_APPROACH] = 0.02
        B[APPROACHING, INTERACTING, ACTION_APPROACH] = 0.08
        B[INTERACTING, INTERACTING, ACTION_APPROACH] = 0.85
        B[DISENGAGING, INTERACTING, ACTION_APPROACH] = 0.05

        # ACTION: interact_led / interact_wiggle (similar effects)
        for action in [ACTION_INTERACT_LED, ACTION_INTERACT_WIGGLE]:
            # Keep interacting if already there
            B[SEARCHING, INTERACTING, action] = 0.02
            B[APPROACHING, INTERACTING, action] = 0.08
            B[INTERACTING, INTERACTING, action] = 0.85
            B[DISENGAGING, INTERACTING, action] = 0.05
            # From approaching, interaction might engage
            B[SEARCHING, APPROACHING, action] = 0.05
            B[APPROACHING, APPROACHING, action] = 0.40
            B[INTERACTING, APPROACHING, action] = 0.50
            B[DISENGAGING, APPROACHING, action] = 0.05
            # From searching, probably not effective
            B[SEARCHING, SEARCHING, action] = 0.60
            B[APPROACHING, SEARCHING, action] = 0.30
            B[INTERACTING, SEARCHING, action] = 0.05
            B[DISENGAGING, SEARCHING, action] = 0.05

        # ACTION: backoff
        # Usually leads to disengaging or searching
        B[SEARCHING, INTERACTING, ACTION_BACKOFF] = 0.10
        B[APPROACHING, INTERACTING, ACTION_BACKOFF] = 0.30
        B[INTERACTING, INTERACTING, ACTION_BACKOFF] = 0.30
        B[DISENGAGING, INTERACTING, ACTION_BACKOFF] = 0.30
        # From approaching
        B[SEARCHING, APPROACHING, ACTION_BACKOFF] = 0.20
        B[APPROACHING, APPROACHING, ACTION_BACKOFF] = 0.40
        B[INTERACTING, APPROACHING, ACTION_BACKOFF] = 0.10
        B[DISENGAGING, APPROACHING, ACTION_BACKOFF] = 0.30

        # ACTION: land
        # End state - transition to disengaging
        for from_state in range(N_ENGAGEMENT_STATES):
            B[SEARCHING, from_state, ACTION_LAND] = 0.05
            B[APPROACHING, from_state, ACTION_LAND] = 0.05
            B[INTERACTING, from_state, ACTION_LAND] = 0.05
            B[DISENGAGING, from_state, ACTION_LAND] = 0.85

        # Normalize columns for each action
        for a in range(N_ACTIONS):
            col_sums = B[:, :, a].sum(axis=0, keepdims=True)
            B[:, :, a] = B[:, :, a] / (col_sums + 1e-10)

        return jnp.array(B)

    def _build_C_preferences(self) -> jnp.ndarray:
        """
        Build observation preference vector C.

        Higher values = more preferred observations.
        We prefer detecting people, especially close.

        Returns:
            C: Shape (N_PERSON_OBS,) - normalized preferences
        """
        # Preferences for each observation type
        C = np.array([
            0.1,   # not_detected - least preferred
            0.5,   # detected_left - okay
            0.7,   # detected_center - good
            0.5,   # detected_right - okay
            1.0,   # detected_close - best (interaction goal)
        ])

        # Normalize to probability distribution
        C = C / C.sum()

        return jnp.array(C)

    def _person_obs_from_token(self, obs: ObservationToken) -> int:
        """
        Convert observation token to person observation index.

        Args:
            obs: Observation token from YOLO

        Returns:
            obs_idx: Index into PERSON_OBS
        """
        if hasattr(obs, 'person_obs_idx'):
            return obs.person_obs_idx

        if not obs.person_detected:
            return 0  # not_detected

        if obs.person_area > 0.5:
            return 4  # detected_close

        cx = obs.person_cx
        if cx < -0.3:
            return 1  # detected_left
        elif cx > 0.3:
            return 3  # detected_right
        else:
            return 2  # detected_center

    def update(self, obs: ObservationToken) -> InteractionResult:
        """
        Update belief and select action based on observation.

        Args:
            obs: Current observation token (with person detection)

        Returns:
            result: InteractionResult with belief, selected action, and diagnostics
        """
        # Get person observation index
        obs_idx = self._person_obs_from_token(obs)

        # Predict step (if we have last action)
        if self._last_action is not None:
            predicted_belief = predict_next_belief(
                self._belief, self._B, self._last_action
            )
        else:
            predicted_belief = self._belief

        # Update step: incorporate observation
        likelihood = self._A[obs_idx, :]
        posterior = belief_update(predicted_belief, likelihood)
        self._belief = posterior

        # Select action via EFE
        efe_values = compute_all_efe(
            self._belief, self._A, self._B, self._C, N_ACTIONS
        )

        # Softmax over negative EFE (lower EFE = higher probability)
        action_probs = softmax(-efe_values, self.temperature)
        selected_action_idx = int(jnp.argmin(efe_values))

        # Update tracking
        self._last_action = selected_action_idx
        most_likely_state = int(jnp.argmax(posterior))
        self._frames_in_state[most_likely_state] += 1
        self._total_actions[selected_action_idx] += 1

        return InteractionResult(
            belief=posterior,
            engagement_state=ENGAGEMENT_STATES[most_likely_state],
            confidence=float(posterior[most_likely_state]),
            selected_action=ACTIONS[selected_action_idx],
            selected_action_idx=selected_action_idx,
            action_probs=action_probs,
            efe_values=efe_values,
            belief_entropy=float(entropy(posterior)),
        )

    def update_with_action_override(
        self,
        obs: ObservationToken,
        override_action: Optional[int] = None
    ) -> InteractionResult:
        """
        Update belief and optionally override action selection.

        Useful when safety module or user overrides the action.

        Args:
            obs: Current observation token
            override_action: If provided, use this action instead of EFE selection

        Returns:
            result: InteractionResult
        """
        result = self.update(obs)

        if override_action is not None and 0 <= override_action < N_ACTIONS:
            # Update last action to the override (for next prediction step)
            self._last_action = override_action
            self._total_actions[override_action] += 1

            # Return modified result
            return InteractionResult(
                belief=result.belief,
                engagement_state=result.engagement_state,
                confidence=result.confidence,
                selected_action=ACTIONS[override_action],
                selected_action_idx=override_action,
                action_probs=result.action_probs,
                efe_values=result.efe_values,
                belief_entropy=result.belief_entropy,
            )

        return result

    def get_action_for_state(self, target_state: int) -> int:
        """
        Get best action to reach a target engagement state.

        Uses current belief and EFE, but with modified preferences
        to favor observations typical of target state.

        Args:
            target_state: Target engagement state index

        Returns:
            action_idx: Best action to reach target
        """
        # Create preference vector favoring target state's typical observations
        target_A = self._A[:, target_state]  # P(o | target_state)
        target_C = normalize(target_A + 0.1)  # Use as preferences

        # Compute EFE with target preferences
        efe_values = compute_all_efe(
            self._belief, self._A, self._B, target_C, N_ACTIONS
        )

        return int(jnp.argmin(efe_values))

    def set_engagement_belief(self, state: int, confidence: float = 0.8) -> None:
        """
        Manually set belief to specific engagement state.

        Useful for resetting after safety events or mode changes.

        Args:
            state: Target state index
            confidence: Probability mass for target state
        """
        remaining = (1.0 - confidence) / (N_ENGAGEMENT_STATES - 1)
        belief = np.ones(N_ENGAGEMENT_STATES) * remaining
        belief[state] = confidence
        self._belief = jnp.array(belief)

    def reset_to_searching(self) -> None:
        """Reset belief to searching mode."""
        self.set_engagement_belief(SEARCHING, 0.7)
        self._last_action = None

    def reset_to_approaching(self) -> None:
        """Reset belief to approaching mode (person just detected)."""
        self.set_engagement_belief(APPROACHING, 0.7)

    def get_engagement_probs(self) -> Dict[str, float]:
        """Get named probabilities for each engagement state."""
        return {
            ENGAGEMENT_STATES[i]: float(self._belief[i])
            for i in range(N_ENGAGEMENT_STATES)
        }

    def get_statistics(self) -> Dict:
        """Get summary statistics for debugging."""
        belief = np.array(self._belief)

        return {
            'engagement_probs': self.get_engagement_probs(),
            'most_likely_state': ENGAGEMENT_STATES[int(np.argmax(belief))],
            'confidence': float(belief.max()),
            'belief_entropy': float(entropy(self._belief)),
            'last_action': ACTIONS[self._last_action] if self._last_action is not None else None,
            'frames_per_state': {
                ENGAGEMENT_STATES[k]: v for k, v in self._frames_in_state.items()
            },
            'action_counts': {
                ACTIONS[k]: v for k, v in self._total_actions.items()
            },
        }

    def save_state(self) -> Dict:
        """Save state for persistence."""
        return {
            'belief': self._belief.tolist(),
            'last_action': self._last_action,
            'frames_in_state': self._frames_in_state,
            'total_actions': self._total_actions,
            'prior_alpha': self.prior_alpha,
            'temperature': self.temperature,
        }

    @classmethod
    def load_state(cls, state: Dict) -> 'InteractionModePOMDP':
        """Load from saved state."""
        pomdp = cls(
            prior_alpha=state.get('prior_alpha', DIRICHLET_PRIOR_ALPHA),
            temperature=state.get('temperature', ACTION_TEMPERATURE),
        )

        if 'belief' in state:
            pomdp._belief = jnp.array(state['belief'])

        pomdp._last_action = state.get('last_action')
        pomdp._frames_in_state = state.get('frames_in_state', {i: 0 for i in range(N_ENGAGEMENT_STATES)})
        pomdp._total_actions = state.get('total_actions', {i: 0 for i in range(N_ACTIONS)})

        return pomdp


# =============================================================================
# Utility Functions
# =============================================================================

def action_to_rc_control(
    action_idx: int,
    person_cx: float = 0.0,
    person_area: float = 0.0,
    max_fb: int = 20,
    max_yaw: int = 25,
    search_yaw: int = 35,
) -> Tuple[int, int, int, int]:
    """
    Convert action index to RC control values.

    Args:
        action_idx: Selected action index
        person_cx: Person center x (-1 to 1, 0 = center)
        person_area: Person bounding box area (0-1)
        max_fb: Max forward/back speed
        max_yaw: Max yaw rate
        search_yaw: Yaw rate during search

    Returns:
        (lr, fb, ud, yaw): RC control tuple
    """
    if action_idx == ACTION_CONTINUE_SEARCH:
        # Rotate to search for person
        return (0, 0, 0, search_yaw)

    elif action_idx == ACTION_APPROACH:
        # Move toward person
        # Yaw to center person
        yaw = int(-person_cx * max_yaw)
        # Move forward (faster if person is small/far)
        fb = int(max_fb * (1.0 - person_area))
        return (0, fb, 0, yaw)

    elif action_idx == ACTION_INTERACT_LED:
        # Stay in place, maybe slight yaw to track
        yaw = int(-person_cx * max_yaw * 0.5)
        return (0, 0, 0, yaw)

    elif action_idx == ACTION_INTERACT_WIGGLE:
        # Wiggle motion (handled externally, just return centering)
        yaw = int(-person_cx * max_yaw * 0.5)
        return (0, 0, 0, yaw)

    elif action_idx == ACTION_BACKOFF:
        # Move backward
        return (0, -max_fb, 0, 0)

    elif action_idx == ACTION_LAND:
        # Landing is handled externally
        return (0, 0, 0, 0)

    else:
        # Unknown action - stop
        return (0, 0, 0, 0)
