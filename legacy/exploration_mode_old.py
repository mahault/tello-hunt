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

        # Door-seeking state: track depth during rotation to find openings
        self._door_scan_active: bool = False
        self._door_scan_depths: list = []  # (yaw_estimate, mean_depth) pairs
        self._door_scan_rotations: int = 0
        self._door_target_direction: Optional[int] = None  # turns needed to face door
        self._current_yaw_estimate: float = 0.0  # track yaw during scan

    def update(
        self,
        obs: 'ObservationToken',
        world_model: 'WorldModel',
        loc_result: 'LocalizationResult',
        depth_map: Optional[np.ndarray] = None,
    ) -> ExplorationResult:
        """
        Update exploration state and select action.

        Args:
            obs: Current observation token
            world_model: WorldModel for EFE-based action selection
            loc_result: Result from WorldModel.localize() (contains VFE)
            depth_map: Optional depth map for door-seeking behavior

        Returns:
            ExplorationResult with action and diagnostics
        """
        self._total_frames += 1
        self._frames_in_state += 1
        self._current_depth_map = depth_map  # Store for door-seeking

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

        IMPORTANT: If CSCG returns near-uniform probabilities, fall back to
        random exploration since the model hasn't learned meaningful transitions.
        """
        n_locations = world_model.n_locations
        debug = (self._total_frames % 30 == 0)  # Debug every 30 frames

        # Track if we're stuck - either at same place OR oscillating between 2-3 places
        if not hasattr(self, '_recent_places'):
            self._recent_places = []  # Track last N places
            self._stuck_frames = 0

        current_place = loc_result.token
        self._recent_places.append(current_place)
        if len(self._recent_places) > 30:  # Track last 30 places (~1 sec)
            self._recent_places.pop(0)

        # Check if stuck: oscillating between same 1-3 places
        unique_places = set(self._recent_places[-20:]) if len(self._recent_places) >= 20 else set()
        is_oscillating = len(unique_places) <= 3 and len(self._recent_places) >= 20

        # Don't count as stuck if we're actively door-seeking
        door_seeking_active = getattr(self, '_door_scan_active', False) or \
                              getattr(self, '_door_target_direction', None) is not None

        if is_oscillating and not door_seeking_active:
            self._stuck_frames += 1
        elif not door_seeking_active:
            self._stuck_frames = 0

        # Initialize blocked/stagnation counters
        if not hasattr(self, '_consecutive_blocked'):
            self._consecutive_blocked = 0
        if not hasattr(self, '_rotation_escape_count'):
            self._rotation_escape_count = 0

        # DOOR-SEEKING TRIGGERS:
        # 1. Blocked twice trying to move forward
        # 2. Oscillating between same 3 places for 50+ frames
        # 3. No new places discovered for 100+ frames (stagnation)
        should_seek_door = False
        seek_reason = ""

        if self._consecutive_blocked >= 2:
            should_seek_door = True
            seek_reason = f"blocked {self._consecutive_blocked}x"

        elif self._stuck_frames > 50:
            should_seek_door = True
            places_str = ','.join(str(p) for p in unique_places)
            seek_reason = f"oscillating [{places_str}] for {self._stuck_frames} frames"

        elif self._frames_without_new_location > 100 and n_locations >= 2:
            should_seek_door = True
            seek_reason = f"no new places for {self._frames_without_new_location} frames"

        if should_seek_door:
            if debug:
                print(f"  [DOOR-SEEK] Triggered: {seek_reason}")
            action = self._door_seeking_action(debug=debug)
            return action

        # Phase 1: Bootstrap - CSCG hasn't learned enough yet
        # Need at least 3 locations AND some time to learn transitions
        min_bootstrap_frames = 300  # At least 300 frames before trusting CSCG
        if n_locations < 3 or self._total_frames < min_bootstrap_frames:
            if debug:
                print(f"  [EXPLORE] Bootstrap phase (n_locs={n_locations}, frames={self._total_frames})")
            return self._bootstrap_action(debug=debug)

        # Phase 2: Try CSCG-driven exploration
        if hasattr(world_model, 'get_exploration_target'):
            action_idx, action_probs = world_model.get_exploration_target()

            # Check if probabilities are near-uniform (CSCG hasn't learned)
            # If max prob is < 0.25 (for 5 actions, uniform = 0.20), it's basically random
            max_prob = float(np.max(action_probs))
            prob_range = float(np.max(action_probs) - np.min(action_probs))

            if prob_range < 0.05:  # Less than 5% difference = essentially uniform
                if debug:
                    print(f"  [EXPLORE] CSCG uniform (range={prob_range:.3f}), using systematic exploration")
                return self._systematic_exploration_action(debug=debug)

            # Never use action=0 (stay) during exploration - pick best movement action
            if action_idx == 0:
                # Find best movement action (indices 1-4)
                movement_probs = action_probs[1:5]  # forward, backward, left, right
                action_idx = int(np.argmax(movement_probs)) + 1
                if debug:
                    print(f"  [EXPLORE] CSCG wanted stay, using action={action_idx} instead")
            else:
                if debug:
                    print(f"  [EXPLORE] CSCG-driven: action={action_idx}, max_prob={max_prob:.3f}")

            return int(action_idx)

        # Fallback to random exploration
        return self._random_exploration_action(debug=debug)

    def _door_seeking_action(self, debug: bool = False) -> int:
        """
        Door-seeking behavior: use depth to find openings (doors).

        Strategy:
        1. If not scanning: start a 360° scan, recording depth at each direction
        2. During scan: rotate and record center depth at each step
        3. After full rotation: identify direction with maximum depth (opening)
        4. Turn toward that direction and move forward

        Returns action index.
        """
        # Get current depth from stored depth map
        depth_map = getattr(self, '_current_depth_map', None)

        # Initialize door scan state
        if not hasattr(self, '_door_scan_active'):
            self._door_scan_active = False
            self._door_scan_depths = []
            self._door_scan_rotations = 0
            self._door_target_direction = None

        # Phase 1: Scanning - rotate and collect depth readings
        if not self._door_scan_active and self._door_target_direction is None:
            # Start a new door scan
            self._door_scan_active = True
            self._door_scan_depths = []
            self._door_scan_rotations = 0
            if debug:
                print(f"  [DOOR-SEEK] Starting 360° scan for openings...")

        if self._door_scan_active:
            # Record depth at current direction
            if depth_map is not None:
                # Sample center region depth (where we'd move toward)
                h, w = depth_map.shape[:2] if len(depth_map.shape) > 1 else (depth_map.shape[0], 1)
                center_y, center_x = h // 2, w // 2
                # Sample a vertical strip in the center (where doors would be)
                y1, y2 = max(0, center_y - h//4), min(h, center_y + h//4)
                x1, x2 = max(0, center_x - w//8), min(w, center_x + w//8)
                center_region = depth_map[y1:y2, x1:x2]

                # Use median depth (robust to outliers)
                if center_region.size > 0:
                    # Higher depth = farther = more open
                    median_depth = float(np.median(center_region))
                    self._door_scan_depths.append((self._door_scan_rotations, median_depth))

            self._door_scan_rotations += 1

            # Full rotation takes ~24 steps at 15°/step
            if self._door_scan_rotations >= 24:
                # Scan complete - find best direction
                self._door_scan_active = False

                if len(self._door_scan_depths) > 0:
                    # Find direction with maximum depth (most open)
                    best_idx, best_depth = max(self._door_scan_depths, key=lambda x: x[1])

                    # Calculate how many turns to face that direction
                    # We've done a full rotation, so figure out relative position
                    current_pos = self._door_scan_rotations  # We're at position 24 (back to start)
                    turns_needed = best_idx  # Turn right this many times from start

                    # If it's more than half a rotation, go the other way
                    if turns_needed > 12:
                        self._door_target_direction = -(24 - turns_needed)  # negative = left
                    else:
                        self._door_target_direction = turns_needed  # positive = right

                    if debug:
                        direction = "right" if self._door_target_direction > 0 else "left"
                        print(f"  [DOOR-SEEK] Found opening at rotation {best_idx} (depth={best_depth:.2f}), "
                              f"turning {direction} {abs(self._door_target_direction)} times")
                else:
                    # No depth data - just pick a random direction
                    self._door_target_direction = 6  # Turn right 90°
                    if debug:
                        print(f"  [DOOR-SEEK] No depth data, defaulting to right turn")

                return 4  # One more right turn to start moving toward target

            # Continue scanning - rotate right
            return 4  # turn right

        # Phase 2: Turn toward the detected opening
        if self._door_target_direction is not None and self._door_target_direction != 0:
            if self._door_target_direction > 0:
                self._door_target_direction -= 1
                if debug and self._door_target_direction == 0:
                    print(f"  [DOOR-SEEK] Facing opening, moving forward")
                return 4  # turn right
            else:
                self._door_target_direction += 1
                if debug and self._door_target_direction == 0:
                    print(f"  [DOOR-SEEK] Facing opening, moving forward")
                return 3  # turn left

        # Phase 3: Move toward the opening
        if self._door_target_direction == 0:
            # Reset state and move forward
            self._door_target_direction = None
            self._consecutive_blocked = 0  # Reset blocked counter
            self._rotation_escape_count = 0
            self._stuck_frames = 0  # Reset oscillation counter
            self._frames_without_new_location = 0  # Reset stagnation counter
            if debug:
                print(f"  [DOOR-SEEK] Moving through opening")
            return 1  # forward

        # Fallback: shouldn't reach here, but turn right to scan
        return 4

    def _systematic_exploration_action(self, debug: bool = False) -> int:
        """
        Systematic exploration using a sweep pattern.

        Pattern: Forward until blocked, then turn right 90°, repeat.
        This covers more area than random wandering.
        """
        # Initialize sweep state
        if not hasattr(self, '_sweep_forward_count'):
            self._sweep_forward_count = 0
            self._sweep_turn_count = 0
            self._sweep_direction = 4  # Start turning right

        # Check if we've been blocked
        blocked = getattr(self, '_consecutive_blocked', 0) > 0

        if blocked:
            # We hit a wall - turn to find new direction
            self._sweep_forward_count = 0
            self._sweep_turn_count += 1

            # After 6 turns (full rotation), try the other direction
            if self._sweep_turn_count > 6:
                self._sweep_turn_count = 0
                self._sweep_direction = 3 if self._sweep_direction == 4 else 4

            action = self._sweep_direction
            if debug:
                print(f"  [EXPLORE] Systematic: blocked, turning {'right' if action == 4 else 'left'}")
        else:
            # Not blocked - move forward, but occasionally turn to look around
            self._sweep_forward_count += 1

            # Every 10 forward moves, do a scan (turn left, then right)
            if self._sweep_forward_count % 30 == 10:
                action = 3  # turn left to scan
                if debug:
                    print(f"  [EXPLORE] Systematic: scanning left")
            elif self._sweep_forward_count % 30 == 20:
                action = 4  # turn right to scan
                if debug:
                    print(f"  [EXPLORE] Systematic: scanning right")
            else:
                action = 1  # forward
                if debug:
                    print(f"  [EXPLORE] Systematic: forward (step {self._sweep_forward_count})")

        return action

    def _random_exploration_action(self, debug: bool = False) -> int:
        """
        Random exploration with bias toward forward movement.

        This is used when CSCG hasn't learned meaningful transitions yet.
        Biases toward forward movement to actually explore new areas.
        """
        self._rotation_frames += 1

        # 60% forward, 15% turn left, 15% turn right, 10% backward
        r = np.random.random()
        if r < 0.60:
            action = 1  # forward
        elif r < 0.75:
            action = 3  # turn left
        elif r < 0.90:
            action = 4  # turn right
        else:
            action = 2  # backward

        if debug:
            action_names = ['stay', 'forward', 'backward', 'left', 'right']
            print(f"  [EXPLORE] Random exploration: {action_names[action]}")

        return action

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
        if not hasattr(self, '_rotation_escape_count'):
            self._rotation_escape_count = 0

        if action == 1:  # forward
            if succeeded:
                # Successfully moved forward - reset all escape state
                self._consecutive_blocked = 0
                self._rotation_escape_count = 0
            else:
                self._consecutive_blocked += 1
        elif action == 2:  # backward
            if succeeded:
                # Successfully moved backward - reset escape state
                self._consecutive_blocked = 0
                self._rotation_escape_count = 0

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
        # Systematic exploration state
        self._sweep_forward_count = 0
        self._sweep_turn_count = 0
        self._sweep_direction = 4
        # Stuck detection state (oscillation-aware)
        self._recent_places = []
        self._stuck_frames = 0
        # Rotation escape state
        self._rotation_escape_count = 0
        # Door-seeking state
        self._door_scan_active = False
        self._door_scan_depths = []
        self._door_scan_rotations = 0
        self._door_target_direction = None
        self._current_depth_map = None

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
