"""
Exploration Mode POMDP for systematic environment mapping.

This module provides exploration behavior that prioritizes building
the topological map before hunting for people.

Two-phase design:
1. Bootstrap: Simple exploration to give CSCG initial data
2. CSCG-driven: Delegate to CSCG's EFE-based action selection

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
# Exploration Phase Constants
# =============================================================================

BOOTSTRAP = 0       # Initial exploration before CSCG has learned
CSCG_DRIVEN = 1     # CSCG drives exploration via EFE

# For backwards compatibility with state names
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
        """
        Select action using two-phase approach:
        1. Bootstrap: Simple scan + forward until CSCG has learned
        2. CSCG-driven: Delegate to CSCG's EFE-based exploration
        """
        n_locations = world_model.n_locations
        debug = (self._total_frames % 30 == 0)  # Debug every 30 frames

        # Phase 1: Bootstrap - CSCG hasn't learned enough yet
        if n_locations < 2:
            if debug:
                print(f"  [EXPLORE] Bootstrap phase (n_locs={n_locations})")
            return self._bootstrap_action(debug=debug)

        # Phase 2: CSCG-driven - use EFE-based exploration target
        if hasattr(world_model, 'get_exploration_target'):
            action_idx, reason = world_model.get_exploration_target()
            if debug:
                print(f"  [EXPLORE] CSCG-driven: action={action_idx}, reason={reason}")
            return int(action_idx)

        # Fallback to bootstrap if no CSCG
        return self._bootstrap_action(debug=debug)

    def _bootstrap_action(self, debug: bool = False) -> int:
        """
        Bootstrap exploration: find open paths and move forward.

        Strategy:
        - Try forward first
        - If blocked (tracked via _consecutive_blocked), rotate to find open path
        - Once we find an open direction, move forward

        Returns action index.
        """
        # Initialize blocked counter if needed
        if not hasattr(self, '_consecutive_blocked'):
            self._consecutive_blocked = 0
            self._last_move_succeeded = True

        self._rotation_frames += 1

        # If we were blocked last time, rotate to find open path
        if self._consecutive_blocked > 0:
            if debug:
                print(f"  [BOOTSTRAP] blocked {self._consecutive_blocked}x, rotating to find path")
            # After rotating, try forward again
            if self._rotation_frames % 3 == 0:  # Every 3rd frame try forward
                return 1  # forward
            return 4  # rotate right

        # Normal exploration: mostly forward with occasional scan
        cycle_pos = self._rotation_frames % 60
        if cycle_pos < 15:
            # Brief scan phase
            if debug:
                print(f"  [BOOTSTRAP] scan phase (cycle {cycle_pos})")
            return 4  # right
        else:
            # Longer move phase - really try to go somewhere
            if debug:
                print(f"  [BOOTSTRAP] move phase (cycle {cycle_pos})")
            return 1  # forward

    def record_movement_result(self, action: int, succeeded: bool):
        """Record whether a movement action succeeded (didn't hit wall)."""
        if not hasattr(self, '_consecutive_blocked'):
            self._consecutive_blocked = 0

        if action == 1:  # forward
            if succeeded:
                self._consecutive_blocked = 0
            else:
                self._consecutive_blocked += 1

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
        """
        Update exploration phase based on CSCG learning progress.

        Simple two-phase logic:
        - Bootstrap (state=0) until CSCG has 2+ locations
        - CSCG-driven (state=1) after that
        """
        n_locations = world_model.n_locations

        if n_locations < 2:
            # Still in bootstrap phase
            if self._state != BOOTSTRAP:
                self._transition_to(BOOTSTRAP)
        else:
            # CSCG has learned enough - it drives exploration
            if self._state != CSCG_DRIVEN:
                self._transition_to(CSCG_DRIVEN)

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
        Check if exploration is complete using model uncertainty.

        Uses CSCG's exploration urgency if available, otherwise falls back to VFE.

        Returns:
            (should_transition, reason)
        """
        n_locs = world_model.n_locations

        # Need minimum exploration time
        min_frames = 200
        if self._total_frames < min_frames:
            return False, f"Exploring ({self._total_frames}/{min_frames} frames)"

        # Check CSCG's exploration urgency if available
        if hasattr(world_model, 'get_exploration_urgency'):
            urgency, urgency_reason = world_model.get_exploration_urgency()
            if urgency > 0.3:
                return False, urgency_reason
            # Model is well-explored according to CSCG
            return True, f"Exploration complete: {urgency_reason}"

        # Fallback to VFE-based check
        if n_locs < 2:
            return False, f"Need more locations ({n_locs}/2)"

        if len(self._vfe_history) < self.vfe_window // 2:
            return False, f"Gathering VFE data ({len(self._vfe_history)}/{self.vfe_window})"

        mean_vfe, vfe_var = self.get_vfe_stats()

        if mean_vfe > self.vfe_threshold:
            return False, f"VFE too high ({mean_vfe:.2f} > {self.vfe_threshold})"

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
        self._consecutive_blocked = 0

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
