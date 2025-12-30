# pomdp/look_ahead_explorer.py
"""
Look-Ahead Exploration Policy with Heading Evaluation.

Key insight: Rotation is a SENSING action, not a movement substitute.
Before moving forward, evaluate multiple candidate headings and pick the best one.

Strategy:
1. Sample candidate headings (current ± {0°, 30°, 60°, 90°})
2. For each heading, compute:
   - S(h) = Safety score (depth-based forward corridor check)
   - N(h) = Novelty score (unexplored directions from topological map)
   - G(h) = Goal progress (distance to frontier/unexplored areas)
3. Pick heading that maximizes: Score(h) = w_safe*S + w_nov*N + w_goal*G - w_turn*|turn|
4. Execute: rotate to best heading, then move forward

This avoids:
- Oscillation (turning becomes purposeful, not reactive)
- Getting stuck (always pick a safe direction)
- Circling in same area (frontier/progress objectives)
"""

import math
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

# Action indices matching the simulator
ROTATE_LEFT = 3
ROTATE_RIGHT = 4
FORWARD = 1
BACKWARD = 2
STRAFE_LEFT = 5
STRAFE_RIGHT = 6

# Rotation step size in radians (simulator uses ~15° per rotate action)
ROTATE_STEP = math.radians(15)


@dataclass
class HeadingScore:
    """Score components for a candidate heading."""
    heading: float  # Absolute heading in radians
    relative: float  # Relative to current heading (radians)

    safety: float  # S(h): 0=blocked, 1=clear corridor
    novelty: float  # N(h): How unexplored this direction is
    goal: float  # G(h): Progress toward frontier/goal
    turn_cost: float  # |h - h_current| normalized

    total: float  # Final combined score

    def __repr__(self):
        return (f"HeadingScore(rel={math.degrees(self.relative):+.0f}°, "
                f"S={self.safety:.2f}, N={self.novelty:.2f}, G={self.goal:.2f}, "
                f"turn={self.turn_cost:.2f}, total={self.total:.2f})")


@dataclass
class ExplorationState:
    """Current exploration state."""
    phase: str  # 'sensing', 'rotating', 'moving', 'escaping'
    target_heading: Optional[float] = None
    steps_remaining: int = 0
    escape_direction: int = BACKWARD


class LookAheadExplorer:
    """
    Exploration policy that evaluates headings before moving.

    Two-phase behavior:
    1. SENSING: Evaluate candidate headings, pick best one
    2. ROTATING: Turn to face the best heading
    3. MOVING: Move forward until blocked or new decision needed
    4. ESCAPING: When stuck, back up and try new direction
    """

    def __init__(
        self,
        # Scoring weights
        w_safety: float = 2.0,  # Safety is most important
        w_novelty: float = 1.5,  # Then novelty
        w_goal: float = 1.0,  # Then goal progress
        w_turn: float = 0.3,  # Small penalty for large turns

        # Heading candidates (relative to current, in degrees)
        heading_candidates: List[float] = None,

        # Thresholds
        min_safety_to_move: float = 0.4,  # Don't move if safety < this
        forward_steps_per_decision: int = 3,  # How many forwards before re-evaluate

        # Escape parameters
        escape_backup_steps: int = 3,
        escape_turn_steps: int = 6,

        # Hallway prior (encourage committing through corridors)
        hallway_progress_weight: float = 0.5,
        hallway_threshold: float = 0.7,  # Safety threshold for hallway
    ):
        self.w_safety = w_safety
        self.w_novelty = w_novelty
        self.w_goal = w_goal
        self.w_turn = w_turn

        # Default candidate headings: 0°, ±30°, ±60°, ±90°, ±180°
        if heading_candidates is None:
            heading_candidates = [0, 30, -30, 60, -60, 90, -90, 180]
        self.heading_candidates = [math.radians(h) for h in heading_candidates]

        self.min_safety_to_move = min_safety_to_move
        self.forward_steps_per_decision = forward_steps_per_decision
        self.escape_backup_steps = escape_backup_steps
        self.escape_turn_steps = escape_turn_steps
        self.hallway_progress_weight = hallway_progress_weight
        self.hallway_threshold = hallway_threshold

        # State
        self.state = ExplorationState(phase='sensing')
        self.current_heading: float = 0.0
        self.forward_count: int = 0
        self.blocked_count: int = 0

        # History for novelty computation
        self.heading_history: List[float] = []
        self.heading_history_size: int = 50
        self.position_history: List[Tuple[float, float, float]] = []  # (x, y, z)

        # Last depth map for corridor analysis
        self._last_depth_map: Optional[np.ndarray] = None

        # Frontier info from spatial mapper
        self._frontier_direction: Optional[float] = None
        self._frontier_distance: float = float('inf')

        # Debug
        self._last_scores: List[HeadingScore] = []
        self._frame_count: int = 0

    def update_heading(self, heading: float):
        """Update current heading from simulator."""
        self.current_heading = heading

    def update_depth(self, depth_map: np.ndarray):
        """Update depth map for safety analysis."""
        self._last_depth_map = depth_map

    def update_position(self, x: float, y: float, z: float):
        """Track position history for novelty."""
        self.position_history.append((x, y, z))
        if len(self.position_history) > 100:
            self.position_history.pop(0)

    def update_frontier(self, direction: Optional[float], distance: float):
        """Update frontier info from spatial mapper."""
        self._frontier_direction = direction
        self._frontier_distance = distance

    def record_block(self, was_blocked: bool):
        """Record if last action was blocked."""
        if was_blocked:
            self.blocked_count += 1
            self.forward_count = 0
        else:
            self.blocked_count = 0
            self.forward_count += 1

    def _compute_safety(self, heading: float, depth_map: Optional[np.ndarray]) -> float:
        """
        Compute safety score for a heading using depth map.

        Checks the forward corridor in that direction.
        Returns 0.0 (blocked) to 1.0 (clear corridor).
        """
        if depth_map is None:
            return 0.5  # Unknown, assume moderate safety

        h, w = depth_map.shape

        # Convert heading to image column
        # Heading is relative to current view (which depth_map shows)
        # Depth map shows what's in front of current heading
        # So we need to map the relative heading offset to image column

        rel_heading = self._normalize_angle(heading - self.current_heading)

        # FOV is approximately 60°, so ±30° maps to image edges
        fov_rad = math.radians(60)

        # If heading is outside current FOV, we can't assess it
        if abs(rel_heading) > fov_rad / 2:
            return 0.5  # Unknown

        # Map heading to column
        # rel_heading = 0 -> center column (w/2)
        # rel_heading = +fov/2 -> right edge (w)
        # rel_heading = -fov/2 -> left edge (0)
        col_frac = 0.5 + (rel_heading / fov_rad)
        center_col = int(col_frac * w)

        # Sample a corridor: center column ± some width
        corridor_width = w // 6  # ~17% of image width
        col_start = max(0, center_col - corridor_width)
        col_end = min(w, center_col + corridor_width)

        # Focus on middle rows (floor level obstacles)
        row_start = int(h * 0.3)
        row_end = int(h * 0.8)

        # Extract corridor region
        corridor = depth_map[row_start:row_end, col_start:col_end]

        if corridor.size == 0:
            return 0.5

        # Depth map: higher values = closer obstacles
        # We want to know if the path is clear

        # Use percentile for robustness
        max_depth = np.percentile(corridor, 90)  # Closest obstacle

        # Convert to safety: high depth (close) = low safety
        # Assume depth is normalized 0-1 where 1 = very close
        safety = 1.0 - max_depth

        # Bonus for consistent depth (corridor, not edge of obstacle)
        depth_variance = np.std(corridor)
        if depth_variance < 0.1:  # Low variance = clear corridor
            safety = min(1.0, safety + 0.2)

        return np.clip(safety, 0.0, 1.0)

    def _compute_novelty(self, heading: float, topo_map=None, place_id: int = 0) -> float:
        """
        Compute novelty score for a heading.

        High novelty = unexplored direction.
        Low novelty = frequently visited direction.
        """
        # Track heading in history
        self.heading_history.append(heading)
        if len(self.heading_history) > self.heading_history_size:
            self.heading_history.pop(0)

        # Method 1: How often have we faced this direction recently?
        similar_count = 0
        for h in self.heading_history[:-1]:  # Exclude current
            diff = abs(self._normalize_angle(h - heading))
            if diff < math.radians(30):
                similar_count += 1

        heading_novelty = 1.0 - (similar_count / max(1, len(self.heading_history) - 1))

        # Method 2: Use topological map edge statistics
        edge_novelty = 1.0
        if topo_map is not None:
            node = topo_map.get_place(place_id)
            if node is not None:
                # Check if this heading corresponds to an untried or unexplored edge
                # Map heading to action (rough approximation)
                rel_heading = self._normalize_angle(heading - self.current_heading)

                if abs(rel_heading) < math.radians(20):
                    # Forward direction
                    edge = node.edges.get(FORWARD)
                    if edge:
                        if edge.attempts == 0:
                            edge_novelty = 1.0
                        else:
                            edge_novelty = 1.0 - edge.blocked_rate

        return 0.6 * heading_novelty + 0.4 * edge_novelty

    def _compute_goal_progress(self, heading: float) -> float:
        """
        Compute goal progress score for a heading.

        Based on distance/direction to frontiers or unexplored areas.
        """
        if self._frontier_direction is None:
            return 0.5  # No frontier info

        # How well does this heading align with frontier direction?
        diff = abs(self._normalize_angle(heading - self._frontier_direction))

        # Score: aligned = 1.0, opposite = 0.0
        alignment = 1.0 - (diff / math.pi)

        # Weight by frontier distance (closer frontier = higher urgency)
        distance_factor = min(1.0, 5.0 / max(0.1, self._frontier_distance))

        return alignment * distance_factor

    def _compute_hallway_progress(self, safety: float, heading: float) -> float:
        """
        Compute hallway progress bonus.

        If we're in a hallway (high safety corridor), encourage committing to it.
        """
        if safety < self.hallway_threshold:
            return 0.0

        # Bonus for continuing straight (small relative turn)
        rel_heading = abs(self._normalize_angle(heading - self.current_heading))

        if rel_heading < math.radians(30):
            # Straight ahead through hallway
            return self.hallway_progress_weight * safety

        return 0.0

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _evaluate_headings(
        self,
        topo_map=None,
        place_id: int = 0,
        debug: bool = False
    ) -> List[HeadingScore]:
        """Evaluate all candidate headings."""
        scores = []

        for rel_heading in self.heading_candidates:
            abs_heading = self._normalize_angle(self.current_heading + rel_heading)

            # Compute score components
            safety = self._compute_safety(abs_heading, self._last_depth_map)
            novelty = self._compute_novelty(abs_heading, topo_map, place_id)
            goal = self._compute_goal_progress(abs_heading)
            hallway = self._compute_hallway_progress(safety, abs_heading)

            # Turn cost: normalized by max turn (180°)
            turn_cost = abs(rel_heading) / math.pi

            # Combined score
            total = (
                self.w_safety * safety +
                self.w_novelty * novelty +
                self.w_goal * goal +
                hallway -
                self.w_turn * turn_cost
            )

            scores.append(HeadingScore(
                heading=abs_heading,
                relative=rel_heading,
                safety=safety,
                novelty=novelty,
                goal=goal,
                turn_cost=turn_cost,
                total=total,
            ))

        # Sort by total score
        scores.sort(key=lambda s: s.total, reverse=True)
        self._last_scores = scores

        if debug:
            print(f"    [LOOK-AHEAD] Heading evaluation (current={math.degrees(self.current_heading):.0f}°):")
            for s in scores[:4]:  # Top 4
                print(f"      {s}")

        return scores

    def choose_action(
        self,
        topo_map=None,
        place_id: int = 0,
        debug: bool = False
    ) -> int:
        """
        Choose the best action using look-ahead heading evaluation.

        Returns action index (1-6).
        """
        self._frame_count += 1
        debug = debug or (self._frame_count % 30 == 0)

        # === ESCAPE MODE ===
        if self.state.phase == 'escaping':
            return self._escape_step(debug)

        # Enter escape mode if too many blocks
        if self.blocked_count >= 3:
            self.state = ExplorationState(
                phase='escaping',
                steps_remaining=self.escape_backup_steps,
                escape_direction=BACKWARD
            )
            if debug:
                print(f"    [LOOK-AHEAD] Entering ESCAPE mode (blocked {self.blocked_count}x)")
            return self._escape_step(debug)

        # === ROTATING PHASE ===
        if self.state.phase == 'rotating':
            return self._rotate_step(debug)

        # === MOVING PHASE ===
        if self.state.phase == 'moving':
            if self.forward_count < self.forward_steps_per_decision:
                if debug:
                    print(f"    [LOOK-AHEAD] Moving forward ({self.forward_count}/{self.forward_steps_per_decision})")
                return FORWARD
            else:
                # Time to re-evaluate
                self.state.phase = 'sensing'
                self.forward_count = 0

        # === SENSING PHASE ===
        scores = self._evaluate_headings(topo_map, place_id, debug)

        if not scores:
            if debug:
                print(f"    [LOOK-AHEAD] No scores, trying forward")
            return FORWARD

        best = scores[0]

        # Check if best heading is safe enough
        if best.safety < self.min_safety_to_move:
            # All directions blocked, enter escape
            self.blocked_count += 1
            if self.blocked_count >= 2:
                self.state = ExplorationState(
                    phase='escaping',
                    steps_remaining=self.escape_backup_steps,
                    escape_direction=BACKWARD
                )
                if debug:
                    print(f"    [LOOK-AHEAD] All directions unsafe (best safety={best.safety:.2f}), escaping")
                return self._escape_step(debug)
            else:
                # Try rotating to best anyway
                pass

        # Compute how many rotation steps needed
        turns_needed = int(round(best.relative / ROTATE_STEP))

        if abs(turns_needed) <= 1:
            # Close enough, move forward
            self.state = ExplorationState(phase='moving')
            if debug:
                print(f"    [LOOK-AHEAD] Aligned, moving forward (safety={best.safety:.2f})")
            return FORWARD
        else:
            # Need to rotate
            self.state = ExplorationState(
                phase='rotating',
                target_heading=best.heading,
                steps_remaining=abs(turns_needed)
            )
            if debug:
                print(f"    [LOOK-AHEAD] Rotating to {math.degrees(best.heading):.0f}° ({turns_needed} steps)")
            return ROTATE_LEFT if turns_needed < 0 else ROTATE_RIGHT

    def _rotate_step(self, debug: bool) -> int:
        """Execute one rotation step."""
        if self.state.target_heading is None:
            self.state.phase = 'sensing'
            return FORWARD

        # Compute remaining turn
        diff = self._normalize_angle(self.state.target_heading - self.current_heading)

        self.state.steps_remaining -= 1

        if abs(diff) < ROTATE_STEP * 0.5 or self.state.steps_remaining <= 0:
            # Close enough
            self.state.phase = 'moving'
            self.forward_count = 0
            if debug:
                print(f"    [LOOK-AHEAD] Rotation complete, now moving")
            return FORWARD

        if diff < 0:
            return ROTATE_LEFT
        else:
            return ROTATE_RIGHT

    def _escape_step(self, debug: bool) -> int:
        """Execute one escape step."""
        self.state.steps_remaining -= 1

        if self.state.escape_direction == BACKWARD:
            if self.state.steps_remaining <= 0:
                # Switch to turning
                self.state.escape_direction = ROTATE_LEFT if np.random.random() < 0.5 else ROTATE_RIGHT
                self.state.steps_remaining = self.escape_turn_steps
                if debug:
                    print(f"    [LOOK-AHEAD] ESCAPE: backup done, now turning")
            else:
                if debug:
                    print(f"    [LOOK-AHEAD] ESCAPE: backing up ({self.state.steps_remaining} left)")
                return BACKWARD

        # Turning phase
        if self.state.steps_remaining <= 0:
            # Done escaping
            self.state.phase = 'sensing'
            self.blocked_count = 0
            if debug:
                print(f"    [LOOK-AHEAD] ESCAPE: complete, resuming exploration")
            return FORWARD

        if debug:
            print(f"    [LOOK-AHEAD] ESCAPE: turning ({self.state.steps_remaining} left)")
        return self.state.escape_direction

    def reset(self):
        """Reset explorer state."""
        self.state = ExplorationState(phase='sensing')
        self.current_heading = 0.0
        self.forward_count = 0
        self.blocked_count = 0
        self.heading_history.clear()
        self.position_history.clear()
        self._last_depth_map = None
        self._frontier_direction = None
        self._frontier_distance = float('inf')
        self._last_scores.clear()
        self._frame_count = 0


class FrontierDetector:
    """
    Detects frontiers (boundaries between explored and unexplored) from occupancy map.

    Used to provide goal direction for the look-ahead explorer.
    """

    def __init__(self, unknown_threshold: int = 127):
        self.unknown_threshold = unknown_threshold

    def find_frontier(
        self,
        occupancy_map: np.ndarray,
        current_pos: Tuple[int, int],
        current_yaw: float
    ) -> Tuple[Optional[float], float]:
        """
        Find nearest frontier direction.

        Args:
            occupancy_map: 2D grid (0=occupied, 127=unknown, 255=free)
            current_pos: (cx, cy) position in map coordinates
            current_yaw: Current heading in radians

        Returns:
            (direction_to_frontier, distance_to_frontier)
            direction is None if no frontier found
        """
        h, w = occupancy_map.shape
        cx, cy = current_pos

        # Find frontier cells (free cells adjacent to unknown)
        free_mask = occupancy_map > self.unknown_threshold + 20
        unknown_mask = np.abs(occupancy_map.astype(int) - self.unknown_threshold) < 20

        # Frontier = free cells with unknown neighbors
        kernel = np.ones((3, 3), dtype=np.uint8)
        unknown_neighbor = cv2.dilate(unknown_mask.astype(np.uint8), kernel)
        frontier_mask = free_mask & (unknown_neighbor > 0)

        # Get frontier cell positions
        frontier_ys, frontier_xs = np.where(frontier_mask)

        if len(frontier_xs) == 0:
            return None, float('inf')

        # Find nearest frontier
        distances = np.sqrt((frontier_xs - cx) ** 2 + (frontier_ys - cy) ** 2)
        nearest_idx = np.argmin(distances)

        nearest_x = frontier_xs[nearest_idx]
        nearest_y = frontier_ys[nearest_idx]
        nearest_dist = distances[nearest_idx]

        # Direction to nearest frontier
        dx = nearest_x - cx
        dy = nearest_y - cy
        direction = math.atan2(dy, dx)

        return direction, nearest_dist


# Need cv2 for FrontierDetector
try:
    import cv2
except ImportError:
    cv2 = None
