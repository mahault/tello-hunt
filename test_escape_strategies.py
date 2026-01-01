"""
Test and compare three escape strategies:
1. Current BACKWARD escape (cooldown-based)
2. Local frontier / free-edge escape (ring scan)
3. Rotate-and-go escape (depth-based heading scan)

Goal: Find which strategy best escapes corner situations.
"""

import sys
import math
import numpy as np
from typing import Tuple, List, Optional
from collections import deque

sys.path.insert(0, '.')

from simulator.glb_simulator import GLBSimulator
from utils.occupancy_map import OccupancyMap


# Action constants
STAY = 0
FORWARD = 1
BACKWARD = 2
ROTATE_LEFT = 3
ROTATE_RIGHT = 4

ACTION_NAMES = ['STAY', 'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']


class EscapeStrategy:
    """Base class for escape strategies."""

    def __init__(self, name: str):
        self.name = name
        self.frame_count = 0
        self.stuck_streak = 0
        self.escape_mode = False
        self.escape_until_frame = 0

    def reset(self):
        self.frame_count = 0
        self.stuck_streak = 0
        self.escape_mode = False
        self.escape_until_frame = 0

    def record_blocked(self):
        """Called when forward movement was blocked."""
        self.stuck_streak += 1

    def record_moved(self):
        """Called when movement succeeded."""
        self.stuck_streak = max(0, self.stuck_streak - 1)

    def choose_action(self, grid: np.ndarray, agent_x: float, agent_y: float,
                     agent_yaw: float, world_to_map_fn, map_to_world_fn,
                     depth_clearance_fn=None) -> int:
        """Choose escape action. Returns action index."""
        raise NotImplementedError


class BackwardEscapeStrategy(EscapeStrategy):
    """
    Current strategy: After 7 consecutive blocks, back up once,
    then cooldown prevents forward for 15 frames.
    """

    def __init__(self):
        super().__init__("BACKWARD")
        self._no_forward_until_frame = 0

    def reset(self):
        super().reset()
        self._no_forward_until_frame = 0

    def choose_action(self, grid: np.ndarray, agent_x: float, agent_y: float,
                     agent_yaw: float, world_to_map_fn, map_to_world_fn,
                     depth_clearance_fn=None) -> int:
        self.frame_count += 1

        # Check if path ahead is blocked
        blocked = False
        for dist in [0.2, 0.4, 0.6]:
            check_x = agent_x + dist * math.cos(agent_yaw)
            check_y = agent_y + dist * math.sin(agent_yaw)
            cx, cy = world_to_map_fn(check_x, check_y)
            if 0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0]:
                if grid[cy, cx] < 50:
                    blocked = True
                    break

        if blocked:
            self.stuck_streak += 1

            # Short-term: alternate turns
            if self.stuck_streak <= 6:
                return ROTATE_LEFT if (self.stuck_streak % 2 == 0) else ROTATE_RIGHT

            # Medium-term: back up
            if self.stuck_streak == 7:
                self._no_forward_until_frame = self.frame_count + 15
                return BACKWARD

            # Long-term: just rotate
            return ROTATE_LEFT if (self.frame_count % 2 == 0) else ROTATE_RIGHT
        else:
            # Path is clear
            if self.frame_count < self._no_forward_until_frame:
                # Cooldown active
                return ROTATE_LEFT if (self.frame_count % 2 == 0) else ROTATE_RIGHT

            self.stuck_streak = 0
            return FORWARD


class LocalFrontierEscapeStrategy(EscapeStrategy):
    """
    Local frontier / free-edge escape:
    When stuck, find reachable cells in a ring around drone,
    score them by traversability/novelty, rotate toward best, move forward.
    """

    def __init__(self):
        super().__init__("LOCAL_FRONTIER")
        self._escape_goal_cell: Optional[Tuple[int, int]] = None
        self._escape_steps_remaining = 0

    def reset(self):
        super().reset()
        self._escape_goal_cell = None
        self._escape_steps_remaining = 0

    def _find_escape_goal(self, grid: np.ndarray, agent_cell: Tuple[int, int],
                         agent_yaw: float, world_to_map_fn, map_to_world_fn) -> Optional[Tuple[int, int]]:
        """
        Find best escape goal cell in a ring around agent.
        Score by: traversability, novelty (unknown adjacency), heading preference.
        """
        ax, ay = agent_cell
        h, w = grid.shape

        best_cell = None
        best_score = -float('inf')

        # Check cells at radii 3-6 cells (about 0.3-0.6m with 0.1m cells)
        for radius in [3, 4, 5, 6]:
            for angle_idx in range(16):
                angle = angle_idx * (2 * math.pi / 16)
                cx = ax + int(round(radius * math.cos(angle)))
                cy = ay + int(round(radius * math.sin(angle)))

                if not (0 <= cx < w and 0 <= cy < h):
                    continue

                cell_val = grid[cy, cx]

                # Skip definite obstacles
                if cell_val < 50:
                    continue

                # Score the cell
                score = 0.0

                # +2 if unknown-ish (exploring value)
                if 100 < cell_val < 180:
                    score += 2.0

                # +1 if free space
                if cell_val > 180:
                    score += 1.0

                # +1 if roughly forward (prefer direction camera can see)
                angle_diff = abs(angle - agent_yaw)
                while angle_diff > math.pi:
                    angle_diff = abs(angle_diff - 2 * math.pi)
                if angle_diff < math.pi / 3:  # Within 60 deg of forward
                    score += 1.0

                # -3 if behind (avoid backward-ish directions)
                if angle_diff > 2 * math.pi / 3:  # More than 120 deg from forward
                    score -= 3.0

                # Check if path to this cell is clear (simple line check)
                path_clear = True
                for step in range(1, radius):
                    sx = ax + int(round(step * math.cos(angle)))
                    sy = ay + int(round(step * math.sin(angle)))
                    if 0 <= sx < w and 0 <= sy < h:
                        if grid[sy, sx] < 50:
                            path_clear = False
                            break

                if not path_clear:
                    score -= 5.0

                if score > best_score:
                    best_score = score
                    best_cell = (cx, cy)

        return best_cell

    def choose_action(self, grid: np.ndarray, agent_x: float, agent_y: float,
                     agent_yaw: float, world_to_map_fn, map_to_world_fn,
                     depth_clearance_fn=None) -> int:
        self.frame_count += 1
        agent_cell = world_to_map_fn(agent_x, agent_y)

        # Check if path ahead is blocked
        blocked = False
        for dist in [0.2, 0.4, 0.6]:
            check_x = agent_x + dist * math.cos(agent_yaw)
            check_y = agent_y + dist * math.sin(agent_yaw)
            cx, cy = world_to_map_fn(check_x, check_y)
            if 0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0]:
                if grid[cy, cx] < 50:
                    blocked = True
                    break

        if blocked:
            self.stuck_streak += 1
        else:
            self.stuck_streak = max(0, self.stuck_streak - 1)

        # Enter escape mode if stuck
        if self.stuck_streak >= 4 and not self.escape_mode:
            self._escape_goal_cell = self._find_escape_goal(
                grid, agent_cell, agent_yaw, world_to_map_fn, map_to_world_fn
            )
            if self._escape_goal_cell is not None:
                self.escape_mode = True
                self._escape_steps_remaining = 5
                self.escape_until_frame = self.frame_count + 20

        # Execute escape
        if self.escape_mode:
            if self._escape_goal_cell is None or self.frame_count > self.escape_until_frame:
                # Escape timeout or no goal
                self.escape_mode = False
                return ROTATE_LEFT

            # Navigate toward escape goal
            goal_x, goal_y = map_to_world_fn(self._escape_goal_cell[0], self._escape_goal_cell[1])
            target_yaw = math.atan2(goal_y - agent_y, goal_x - agent_x)

            heading_error = target_yaw - agent_yaw
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi

            # Check if reached goal
            dist_to_goal = math.sqrt((goal_x - agent_x)**2 + (goal_y - agent_y)**2)
            if dist_to_goal < 0.2:
                self.escape_mode = False
                self.stuck_streak = 0
                return FORWARD

            # Rotate toward goal
            if abs(heading_error) > 0.3:
                return ROTATE_LEFT if heading_error > 0 else ROTATE_RIGHT

            # Move forward
            self._escape_steps_remaining -= 1
            if self._escape_steps_remaining <= 0:
                self.escape_mode = False
            return FORWARD

        # Normal mode - if clear, go forward
        if not blocked:
            return FORWARD
        else:
            return ROTATE_LEFT if (self.frame_count % 2 == 0) else ROTATE_RIGHT


class RotateAndGoEscapeStrategy(EscapeStrategy):
    """
    Rotate-and-go escape using depth/grid only:
    When stuck, try multiple headings relative to current yaw,
    estimate clearance for each, rotate toward best and move forward.
    """

    def __init__(self):
        super().__init__("ROTATE_AND_GO")
        self._best_heading: Optional[float] = None
        self._escape_steps_remaining = 0

    def reset(self):
        super().reset()
        self._best_heading = None
        self._escape_steps_remaining = 0

    def _estimate_clearance(self, grid: np.ndarray, agent_x: float, agent_y: float,
                           heading: float, world_to_map_fn, max_dist: float = 1.0) -> float:
        """Estimate forward clearance at a given heading using grid."""
        clearance = 0.0
        for dist in np.arange(0.1, max_dist + 0.1, 0.1):
            check_x = agent_x + dist * math.cos(heading)
            check_y = agent_y + dist * math.sin(heading)
            cx, cy = world_to_map_fn(check_x, check_y)

            if not (0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0]):
                break

            if grid[cy, cx] < 50:  # Obstacle
                break

            clearance = dist

        return clearance

    def _find_best_heading(self, grid: np.ndarray, agent_x: float, agent_y: float,
                          agent_yaw: float, world_to_map_fn) -> Tuple[float, float]:
        """
        Try headings at various angles relative to current yaw.
        Return (best_heading, clearance).
        """
        test_offsets = [-60, -30, -15, 0, 15, 30, 60]  # degrees
        best_heading = agent_yaw
        best_clearance = 0.0

        for offset_deg in test_offsets:
            test_heading = agent_yaw + math.radians(offset_deg)
            clearance = self._estimate_clearance(grid, agent_x, agent_y, test_heading, world_to_map_fn)

            # Slight preference for smaller turns
            effective_clearance = clearance - abs(offset_deg) * 0.001

            if effective_clearance > best_clearance:
                best_clearance = clearance
                best_heading = test_heading

        return best_heading, best_clearance

    def choose_action(self, grid: np.ndarray, agent_x: float, agent_y: float,
                     agent_yaw: float, world_to_map_fn, map_to_world_fn,
                     depth_clearance_fn=None) -> int:
        self.frame_count += 1

        # Check if path ahead is blocked
        blocked = False
        for dist in [0.2, 0.4, 0.6]:
            check_x = agent_x + dist * math.cos(agent_yaw)
            check_y = agent_y + dist * math.sin(agent_yaw)
            cx, cy = world_to_map_fn(check_x, check_y)
            if 0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0]:
                if grid[cy, cx] < 50:
                    blocked = True
                    break

        if blocked:
            self.stuck_streak += 1
        else:
            self.stuck_streak = max(0, self.stuck_streak - 1)

        # Enter escape mode if stuck
        if self.stuck_streak >= 4 and not self.escape_mode:
            best_heading, clearance = self._find_best_heading(
                grid, agent_x, agent_y, agent_yaw, world_to_map_fn
            )

            if clearance > 0.2:
                self._best_heading = best_heading
                self.escape_mode = True
                self._escape_steps_remaining = 5
                self.escape_until_frame = self.frame_count + 20
            else:
                # No good heading found - just rotate
                return ROTATE_LEFT

        # Execute escape
        if self.escape_mode:
            if self._best_heading is None or self.frame_count > self.escape_until_frame:
                self.escape_mode = False
                return ROTATE_LEFT

            # Rotate toward best heading
            heading_error = self._best_heading - agent_yaw
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi

            if abs(heading_error) > 0.2:
                return ROTATE_LEFT if heading_error > 0 else ROTATE_RIGHT

            # Move forward
            self._escape_steps_remaining -= 1
            if self._escape_steps_remaining <= 0:
                self.escape_mode = False
                self.stuck_streak = 0
            return FORWARD

        # Normal mode
        if not blocked:
            return FORWARD
        else:
            return ROTATE_LEFT if (self.frame_count % 2 == 0) else ROTATE_RIGHT


def run_escape_test(strategy: EscapeStrategy, sim: GLBSimulator,
                   occ_map: OccupancyMap, corner_pos: Tuple[float, float, float],
                   max_frames: int = 60) -> dict:
    """
    Run escape test with a given strategy.

    Args:
        strategy: The escape strategy to test
        sim: GLB simulator instance
        occ_map: Occupancy map with obstacles marked
        corner_pos: (x, z, yaw_deg) to place drone
        max_frames: Maximum frames to run

    Returns:
        Results dictionary with metrics
    """
    # Reset
    strategy.reset()
    sim.x, sim.z = corner_pos[0], corner_pos[1]
    sim.yaw = math.radians(corner_pos[2])

    # Coordinate conversion
    start_x = sim.model_center[0] if hasattr(sim, 'model_center') else -102
    start_z = sim.model_center[2] if hasattr(sim, 'model_center') else 333

    def sim_to_world(sim_x, sim_z):
        world_x = -(sim_z - start_z) / 165.0
        world_y = -(sim_x - start_x) / 165.0
        return world_x, world_y

    positions = []
    actions_taken = []
    stuck_frames = 0

    initial_world_pos = sim_to_world(sim.x, sim.z)

    for frame in range(max_frames):
        world_x, world_y = sim_to_world(sim.x, sim.z)
        world_yaw = sim.yaw

        action = strategy.choose_action(
            grid=occ_map.grid,
            agent_x=world_x,
            agent_y=world_y,
            agent_yaw=world_yaw,
            world_to_map_fn=occ_map.world_to_map,
            map_to_world_fn=occ_map.map_to_world,
        )

        action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else f'ACTION_{action}'

        # Execute action
        old_x, old_z = sim.x, sim.z
        if action > 0:
            moved = sim.move(action, debug=False)
        else:
            moved = True

        # Track if stuck (position didn't change on forward)
        if action == FORWARD and abs(sim.x - old_x) < 0.1 and abs(sim.z - old_z) < 0.1:
            stuck_frames += 1

        positions.append((sim.x, sim.z))
        actions_taken.append(action_name)

    # Calculate metrics
    final_world_pos = sim_to_world(sim.x, sim.z)
    total_distance = math.sqrt(
        (final_world_pos[0] - initial_world_pos[0])**2 +
        (final_world_pos[1] - initial_world_pos[1])**2
    )

    # Count unique positions (measure of exploration)
    unique_positions = len(set((int(p[0]/5), int(p[1]/5)) for p in positions))

    # Check if escaped corner (moved significantly from start)
    escaped = total_distance > 0.3

    return {
        'strategy': strategy.name,
        'total_distance': total_distance,
        'unique_positions': unique_positions,
        'stuck_frames': stuck_frames,
        'escaped': escaped,
        'actions': actions_taken,
        'final_pos': (sim.x, sim.z),
        'action_counts': {a: actions_taken.count(a) for a in set(actions_taken)}
    }


def main():
    print("=" * 70)
    print("ESCAPE STRATEGY COMPARISON TEST")
    print("=" * 70)

    # Initialize simulator
    print("\n1. Loading GLB simulator...")
    sim = GLBSimulator(
        glb_path="simulator/home_-_p3.glb",
        width=640,
        height=480
    )

    # Initialize occupancy map
    print("2. Initializing occupancy map...")
    occ_map = OccupancyMap()

    # Get coordinate conversion
    start_x = sim.model_center[0] if hasattr(sim, 'model_center') else -102
    start_z = sim.model_center[2] if hasattr(sim, 'model_center') else 333

    # Corner test position
    corner_pos = (-298.0, 280.0, 210)  # x, z, yaw_deg (facing into corner)

    # Mark obstacles ahead of corner position
    print("3. Marking obstacles in occupancy grid...")
    world_x = -(corner_pos[1] - start_z) / 165.0
    world_y = -(corner_pos[0] - start_x) / 165.0
    world_yaw = math.radians(corner_pos[2])

    for dist in [0.2, 0.3, 0.4, 0.5, 0.6]:
        obs_x = world_x + math.cos(world_yaw) * dist
        obs_y = world_y + math.sin(world_yaw) * dist
        cx, cy = occ_map.world_to_map(obs_x, obs_y)
        if 0 <= cx < occ_map.grid.shape[1] and 0 <= cy < occ_map.grid.shape[0]:
            occ_map.grid[cy, cx] = 0  # Mark as occupied
            # Also mark adjacent cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    ncx, ncy = cx + dx, cy + dy
                    if 0 <= ncx < occ_map.grid.shape[1] and 0 <= ncy < occ_map.grid.shape[0]:
                        occ_map.grid[ncy, ncx] = 0

    # Also mark obstacles to the sides (true corner)
    for angle_offset in [-0.5, 0.5]:  # Left and right of forward
        side_yaw = world_yaw + angle_offset
        for dist in [0.3, 0.4, 0.5]:
            obs_x = world_x + math.cos(side_yaw) * dist
            obs_y = world_y + math.sin(side_yaw) * dist
            cx, cy = occ_map.world_to_map(obs_x, obs_y)
            if 0 <= cx < occ_map.grid.shape[1] and 0 <= cy < occ_map.grid.shape[0]:
                occ_map.grid[cy, cx] = 0

    print(f"   Corner position: sim=({corner_pos[0]}, {corner_pos[1]}) world=({world_x:.2f}, {world_y:.2f})")

    # Test each strategy
    strategies = [
        BackwardEscapeStrategy(),
        LocalFrontierEscapeStrategy(),
        RotateAndGoEscapeStrategy(),
    ]

    results = []

    print("\n4. Testing escape strategies...")
    print("-" * 70)

    for strategy in strategies:
        print(f"\n   Testing: {strategy.name}")
        result = run_escape_test(strategy, sim, occ_map, corner_pos, max_frames=60)
        results.append(result)

        print(f"   - Distance moved: {result['total_distance']:.3f}m")
        print(f"   - Unique positions: {result['unique_positions']}")
        print(f"   - Stuck frames: {result['stuck_frames']}")
        print(f"   - Escaped: {'YES' if result['escaped'] else 'NO'}")
        print(f"   - Action counts: {result['action_counts']}")
        print(f"   - Final pos: ({result['final_pos'][0]:.1f}, {result['final_pos'][1]:.1f})")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Strategy':<20} {'Distance':>10} {'Unique Pos':>12} {'Stuck':>8} {'Escaped':>10}")
    print("-" * 60)

    for r in results:
        print(f"{r['strategy']:<20} {r['total_distance']:>10.3f}m {r['unique_positions']:>12} {r['stuck_frames']:>8} {'YES' if r['escaped'] else 'NO':>10}")

    # Determine winner
    print("\n" + "-" * 60)

    # Score: prioritize escape, then distance, then fewer stuck frames
    def score(r):
        return (
            (10.0 if r['escaped'] else 0.0) +
            r['total_distance'] * 2.0 +
            r['unique_positions'] * 0.1 -
            r['stuck_frames'] * 0.1
        )

    scored = [(score(r), r) for r in results]
    scored.sort(reverse=True)

    print(f"\nBEST STRATEGY: {scored[0][1]['strategy']} (score: {scored[0][0]:.2f})")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
