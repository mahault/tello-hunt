"""
Exploration Mode POMDP for systematic environment mapping.

This module provides exploration behavior that prioritizes building
the topological map before hunting for people. Uses VFE (Variational
Free Energy) as the signal for exploration completion.

Key principles:
- High VFE / High Surprisal = Observations don't fit model → keep exploring
- Low VFE / Low Surprisal = Observations expected → world is modeled → hunt
"""

from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np
import jax.numpy as jnp

from .config import (
    MOVEMENT_ACTIONS,
    N_MOVEMENT_ACTIONS,
    EXPLORATION_VFE_WINDOW,
    EXPLORATION_VFE_THRESHOLD,
    EXPLORATION_VFE_VARIANCE_THRESHOLD,
    EXPLORATION_EPISTEMIC_WEIGHT,
    EXPLORATION_PRAGMATIC_WEIGHT,
    EXPLORATION_SCAN_YAW,
    EXPLORATION_FORWARD_FB,
    EXPLORATION_STATES,
    N_EXPLORATION_STATES,
)
from .core import entropy

if TYPE_CHECKING:
    from .world_model import WorldModel, LocalizationResult
    from .observation_encoder import ObservationToken


# =============================================================================
# Exploration State Indices
# =============================================================================

SCANNING = 0
APPROACHING_FRONTIER = 1
BACKTRACKING = 2
TRANSITIONING = 3


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class ExplorationResult:
    """Result from exploration mode update."""

    # Current exploration state
    exploration_state: str
    exploration_state_idx: int

    # Selected movement action (0-4: stay, forward, back, left, right)
    selected_action: int
    selected_action_name: str

    # VFE tracking
    current_vfe: float
    mean_vfe: float
    vfe_variance: float

    # Metrics
    locations_discovered: int
    frames_in_exploration: int

    # Transition recommendation
    should_transition_to_hunt: bool
    transition_reason: str


# =============================================================================
# Exploration Mode POMDP
# =============================================================================

class ExplorationModePOMDP:
    """
    POMDP for systematic environment exploration.

    Prioritizes:
    1. Rotating to scan surroundings (epistemic value)
    2. Moving toward unexplored areas (frontier-based)
    3. Building the topological map before hunting

    Uses VFE as the principled metric for determining when
    exploration is complete (model fits observations well).
    """

    def __init__(
        self,
        vfe_window: int = EXPLORATION_VFE_WINDOW,
        vfe_threshold: float = EXPLORATION_VFE_THRESHOLD,
        vfe_variance_threshold: float = EXPLORATION_VFE_VARIANCE_THRESHOLD,
        epistemic_weight: float = EXPLORATION_EPISTEMIC_WEIGHT,
        pragmatic_weight: float = EXPLORATION_PRAGMATIC_WEIGHT,
    ):
        """
        Initialize exploration mode.

        Args:
            vfe_window: Number of frames to track VFE history
            vfe_threshold: Mean VFE threshold for transition
            vfe_variance_threshold: VFE variance threshold for stability
            epistemic_weight: Weight for curiosity/exploration
            pragmatic_weight: Weight for goal achievement
        """
        self.vfe_window = vfe_window
        self.vfe_threshold = vfe_threshold
        self.vfe_variance_threshold = vfe_variance_threshold
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight

        # VFE tracking
        self._vfe_history: deque = deque(maxlen=vfe_window)

        # Exploration state
        self._state: int = SCANNING
        self._frames_in_state: int = 0
        self._total_frames: int = 0

        # Scanning state tracking
        self._rotation_frames: int = 0
        self._rotations_completed: int = 0

        # Stuck detection
        self._frames_without_new_location: int = 0
        self._last_n_locations: int = 0

    def update(
        self,
        obs: 'ObservationToken',
        world_model: 'WorldModel',
        loc_result: 'LocalizationResult',
    ) -> ExplorationResult:
        """
        Update exploration state and select action.

        Args:
            obs: Current observation token
            world_model: WorldModel for EFE-based action selection
            loc_result: Result from WorldModel.localize() (contains VFE)

        Returns:
            ExplorationResult with action and diagnostics
        """
        self._total_frames += 1
        self._frames_in_state += 1

        # Track VFE
        self._vfe_history.append(loc_result.vfe)

        # Track new locations
        n_locs = world_model.n_locations
        if n_locs > self._last_n_locations:
            self._frames_without_new_location = 0
            self._last_n_locations = n_locs
        else:
            self._frames_without_new_location += 1

        # State machine logic
        action_idx = self._select_action(world_model, loc_result)

        # Check for state transitions
        self._update_state(world_model)

        # Check if should transition to hunting
        should_transition, reason = self.should_transition_to_hunt(world_model)

        # Get VFE stats
        mean_vfe, vfe_var = self.get_vfe_stats()

        return ExplorationResult(
            exploration_state=EXPLORATION_STATES[self._state],
            exploration_state_idx=self._state,
            selected_action=action_idx,
            selected_action_name=MOVEMENT_ACTIONS[action_idx],
            current_vfe=loc_result.vfe,
            mean_vfe=mean_vfe,
            vfe_variance=vfe_var,
            locations_discovered=n_locs,
            frames_in_exploration=self._total_frames,
            should_transition_to_hunt=should_transition,
            transition_reason=reason,
        )

    def _select_action(
        self,
        world_model: 'WorldModel',
        loc_result: 'LocalizationResult',
    ) -> int:
        """Select action based on current exploration state."""

        if self._state == SCANNING:
            return self._scanning_action()

        elif self._state == APPROACHING_FRONTIER:
            return self._frontier_action(world_model)

        elif self._state == BACKTRACKING:
            return self._backtrack_action()

        elif self._state == TRANSITIONING:
            # Stay in place during transition
            return 0  # stay

        return 4  # right (default to scanning)

    def _scanning_action(self) -> int:
        """
        Scanning behavior: rotate in place to observe surroundings.

        Returns action index for rotation.
        """
        self._rotation_frames += 1

        # Approximately 90 frames for a full rotation at 40°/s
        # (assuming ~30fps and 360°/40° = 9s per rotation)
        if self._rotation_frames >= 90:
            self._rotation_frames = 0
            self._rotations_completed += 1

        return 4  # right (continuous rotation)

    def _frontier_action(self, world_model: 'WorldModel') -> int:
        """
        Frontier approach: use EFE to select movement toward unexplored areas.

        Args:
            world_model: For get_exploration_target()

        Returns:
            Movement action index
        """
        # Use WorldModel's EFE-based exploration target
        action_idx, _ = world_model.get_exploration_target()
        return int(action_idx)

    def _backtrack_action(self) -> int:
        """
        Backtracking: move backward to previous location.

        Returns action index for backward movement.
        """
        return 2  # back

    def _update_state(self, world_model: 'WorldModel') -> None:
        """Update exploration state based on conditions."""

        if self._state == SCANNING:
            # After completing at least one rotation, consider moving
            if self._rotations_completed >= 1 and self._frames_in_state > 100:
                # If we have locations and haven't found new ones recently,
                # try moving to a frontier
                if world_model.n_locations >= 1:
                    self._transition_to(APPROACHING_FRONTIER)

        elif self._state == APPROACHING_FRONTIER:
            # If stuck (no new locations for many frames), try backtracking
            if self._frames_without_new_location > 150:
                self._transition_to(BACKTRACKING)

            # If we've been approaching for a while, go back to scanning
            if self._frames_in_state > 200:
                self._transition_to(SCANNING)

        elif self._state == BACKTRACKING:
            # After backtracking for a bit, go back to scanning
            if self._frames_in_state > 60:
                self._transition_to(SCANNING)

    def _transition_to(self, new_state: int) -> None:
        """Transition to a new exploration state."""
        self._state = new_state
        self._frames_in_state = 0

        if new_state == SCANNING:
            self._rotation_frames = 0

    def should_transition_to_hunt(
        self,
        world_model: 'WorldModel',
    ) -> Tuple[bool, str]:
        """
        Check if exploration is complete using VFE metrics.

        Transition when:
        1. Mean VFE is low (observations fit the model)
        2. VFE variance is low (stable, not fluctuating)
        3. At least one location has been established

        Returns:
            (should_transition, reason)
        """
        n_locs = world_model.n_locations

        # Need at least one location
        if n_locs < 1:
            return False, "No locations discovered yet"

        # Need enough VFE history
        if len(self._vfe_history) < self.vfe_window // 2:
            return False, f"Gathering VFE data ({len(self._vfe_history)}/{self.vfe_window})"

        mean_vfe, vfe_var = self.get_vfe_stats()

        # Check VFE threshold
        if mean_vfe > self.vfe_threshold:
            return False, f"VFE too high ({mean_vfe:.2f} > {self.vfe_threshold})"

        # Check variance threshold
        if vfe_var > self.vfe_variance_threshold:
            return False, f"VFE unstable (var={vfe_var:.2f} > {self.vfe_variance_threshold})"

        return True, f"Model stable (VFE={mean_vfe:.2f}, var={vfe_var:.2f}, locs={n_locs})"

    def get_vfe_stats(self) -> Tuple[float, float]:
        """
        Get VFE statistics from history.

        Returns:
            (mean_vfe, variance)
        """
        if len(self._vfe_history) == 0:
            return float('inf'), float('inf')

        vfe_array = np.array(list(self._vfe_history))
        mean_vfe = float(np.mean(vfe_array))
        vfe_var = float(np.var(vfe_array))

        return mean_vfe, vfe_var

    def reset(self) -> None:
        """Reset exploration state for a new exploration session."""
        self._vfe_history.clear()
        self._state = SCANNING
        self._frames_in_state = 0
        self._total_frames = 0
        self._rotation_frames = 0
        self._rotations_completed = 0
        self._frames_without_new_location = 0
        self._last_n_locations = 0

    def get_statistics(self) -> dict:
        """Get exploration statistics."""
        mean_vfe, vfe_var = self.get_vfe_stats()
        return {
            'state': EXPLORATION_STATES[self._state],
            'total_frames': self._total_frames,
            'rotations_completed': self._rotations_completed,
            'mean_vfe': mean_vfe,
            'vfe_variance': vfe_var,
            'vfe_history_length': len(self._vfe_history),
        }

    def __repr__(self) -> str:
        mean_vfe, vfe_var = self.get_vfe_stats()
        return (f"ExplorationModePOMDP(state={EXPLORATION_STATES[self._state]}, "
                f"frames={self._total_frames}, vfe={mean_vfe:.2f})")


# =============================================================================
# RC Control Conversion
# =============================================================================

def exploration_action_to_rc_control(
    action_idx: int,
    scan_yaw: int = EXPLORATION_SCAN_YAW,
    explore_fb: int = EXPLORATION_FORWARD_FB,
) -> Tuple[int, int, int, int]:
    """
    Convert exploration action to RC control values.

    Different from hunting action_to_rc_control:
    - Faster rotation for scanning
    - More forward movement for frontier exploration

    Args:
        action_idx: Movement action (0-4)
        scan_yaw: Rotation speed during exploration
        explore_fb: Forward speed during exploration

    Returns:
        (lr, fb, ud, yaw) RC control tuple
    """
    if action_idx == 0:  # stay
        # During exploration, "stay" means rotate to scan
        return (0, 0, 0, scan_yaw)

    elif action_idx == 1:  # forward
        return (0, explore_fb, 0, 0)

    elif action_idx == 2:  # back
        return (0, -explore_fb, 0, 0)

    elif action_idx == 3:  # left
        return (0, 0, 0, -scan_yaw)

    elif action_idx == 4:  # right
        return (0, 0, 0, scan_yaw)

    return (0, 0, 0, 0)  # Default: stop
