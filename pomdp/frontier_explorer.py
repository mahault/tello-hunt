# pomdp/frontier_explorer.py
"""
Frontier-Based Exploration for Tello Drone.

The canonical solution for exploring unknown connected environments:
1. Find frontiers (free cells adjacent to unknown)
2. Select nearest frontier
3. Navigate toward it
4. Repeat until no frontiers remain

No oscillation penalties. No escape mode. No special hallway logic.
Rooms are discovered naturally as the agent is pulled through doorways
toward the unexplored space behind them.
"""

import numpy as np
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Frontier:
    """A frontier cell on the boundary between explored and unexplored."""
    cell_x: int
    cell_y: int
    world_x: float
    world_y: float
    distance: float  # Distance from agent


# Action indices matching the simulator
STAY = 0
FORWARD = 1
BACKWARD = 2
ROTATE_LEFT = 3
ROTATE_RIGHT = 4


class FrontierExplorer:
    """
    Frontier-based exploration policy.

    Algorithm:
        IF no current_target OR current_target reached OR invalid:
            frontiers = extract_frontiers(map)
            IF frontiers empty:
                DONE
            current_target = nearest frontier

        navigate_toward(current_target)

    That's it.
    """

    def __init__(
        self,
        reached_threshold: float = 0.3,    # Meters - consider frontier reached
        heading_tolerance: float = 0.35,   # Radians (~20 deg) - when to move forward
        min_frontier_distance: float = 0.2,  # Ignore frontiers closer than this
    ):
        self.reached_threshold = reached_threshold
        self.heading_tolerance = heading_tolerance
        self.min_frontier_distance = min_frontier_distance

        # Current target
        self.target: Optional[Tuple[float, float]] = None  # (world_x, world_y)
        self.target_cell: Optional[Tuple[int, int]] = None

        # State
        self.frontiers: List[Frontier] = []
        self.exploration_complete = False
        self._frame_count = 0

    def extract_frontiers(
        self,
        grid: np.ndarray,
        unknown_val: int = 127,
        free_threshold: int = 180,  # Cells above this are considered free
    ) -> List[Tuple[int, int]]:
        """
        Extract frontier cells from occupancy grid.

        A frontier is a FREE cell adjacent to an UNKNOWN cell.

        Args:
            grid: Occupancy grid (H x W), values 0-255
            unknown_val: Value representing unknown space
            free_threshold: Values above this are considered free

        Returns:
            List of (cell_x, cell_y) frontier positions
        """
        h, w = grid.shape
        frontiers = []

        # Masks
        free_mask = grid > free_threshold
        unknown_mask = np.abs(grid.astype(np.int16) - unknown_val) < 20

        # 4-connectivity neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # For each free cell, check if any neighbor is unknown
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if free_mask[y, x]:
                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if unknown_mask[ny, nx]:
                            frontiers.append((x, y))
                            break  # Only add once

        return frontiers

    def select_nearest_frontier(
        self,
        frontiers: List[Tuple[int, int]],
        agent_cell: Tuple[int, int],
        map_to_world_fn,
    ) -> Optional[Frontier]:
        """
        Select the nearest frontier to the agent.

        Args:
            frontiers: List of (cell_x, cell_y) frontier positions
            agent_cell: Agent's current cell position
            map_to_world_fn: Function to convert cell to world coordinates

        Returns:
            Nearest Frontier, or None if no frontiers
        """
        if not frontiers:
            return None

        ax, ay = agent_cell
        agent_world = map_to_world_fn(ax, ay)

        nearest = None
        min_dist = float('inf')

        for cx, cy in frontiers:
            # Euclidean distance in cells
            dist_cells = math.sqrt((cx - ax) ** 2 + (cy - ay) ** 2)

            # Convert to world distance
            wx, wy = map_to_world_fn(cx, cy)
            dist_world = math.sqrt((wx - agent_world[0]) ** 2 + (wy - agent_world[1]) ** 2)

            if dist_world < self.min_frontier_distance:
                continue  # Too close, skip

            if dist_world < min_dist:
                min_dist = dist_world
                nearest = Frontier(
                    cell_x=cx, cell_y=cy,
                    world_x=wx, world_y=wy,
                    distance=dist_world
                )

        return nearest

    def compute_heading_to_target(
        self,
        agent_x: float,
        agent_y: float,
        target_x: float,
        target_y: float,
    ) -> float:
        """Compute heading (radians) from agent to target."""
        dx = target_x - agent_x
        dy = target_y - agent_y
        return math.atan2(dy, dx)

    def choose_action(
        self,
        grid: np.ndarray,
        agent_x: float,
        agent_y: float,
        agent_yaw: float,
        world_to_map_fn,
        map_to_world_fn,
        debug: bool = False,
    ) -> int:
        """
        Choose action using frontier-based exploration.

        Args:
            grid: Occupancy grid
            agent_x, agent_y: Agent world position (meters)
            agent_yaw: Agent heading (radians)
            world_to_map_fn: Function to convert world to cell coordinates
            map_to_world_fn: Function to convert cell to world coordinates
            debug: Print debug info

        Returns:
            Action index (0-4)
        """
        self._frame_count += 1

        # Get agent cell position
        agent_cell = world_to_map_fn(agent_x, agent_y)

        # Check if we've reached current target
        if self.target is not None:
            dist_to_target = math.sqrt(
                (agent_x - self.target[0]) ** 2 +
                (agent_y - self.target[1]) ** 2
            )
            if dist_to_target < self.reached_threshold:
                if debug:
                    print(f"  [FRONTIER] Reached target at ({self.target[0]:.1f}, {self.target[1]:.1f})")
                self.target = None
                self.target_cell = None

        # Check if current target is still valid (still unknown nearby)
        if self.target_cell is not None:
            tx, ty = self.target_cell
            if 0 <= tx < grid.shape[1] and 0 <= ty < grid.shape[0]:
                # Check if the target area is still a frontier
                if grid[ty, tx] < 150:  # No longer free space
                    if debug:
                        print(f"  [FRONTIER] Target invalidated (no longer free)")
                    self.target = None
                    self.target_cell = None

        # Find new target if needed
        if self.target is None:
            frontiers = self.extract_frontiers(grid)
            self.frontiers = []  # Store for visualization

            if not frontiers:
                # Check if we have any FREE cells at all
                # If not, we haven't made any observations yet - rotate to start
                free_threshold = 180
                has_free_cells = np.any(grid > free_threshold)

                if not has_free_cells:
                    # No observations yet - rotate to build initial map
                    if debug:
                        print(f"  [FRONTIER] No observations yet - rotating to build map")
                    return ROTATE_LEFT

                # We have FREE cells but no frontiers - exploration is complete
                self.exploration_complete = True
                if debug:
                    print(f"  [FRONTIER] No frontiers remaining - exploration complete!")
                return STAY

            nearest = self.select_nearest_frontier(frontiers, agent_cell, map_to_world_fn)

            if nearest is None:
                if debug:
                    print(f"  [FRONTIER] {len(frontiers)} frontiers but none selectable")
                return ROTATE_LEFT  # Spin to discover more

            self.target = (nearest.world_x, nearest.world_y)
            self.target_cell = (nearest.cell_x, nearest.cell_y)

            # Store all frontiers for visualization
            for cx, cy in frontiers:
                wx, wy = map_to_world_fn(cx, cy)
                dist = math.sqrt((wx - agent_x) ** 2 + (wy - agent_y) ** 2)
                self.frontiers.append(Frontier(cx, cy, wx, wy, dist))

            if debug:
                print(f"  [FRONTIER] {len(frontiers)} frontiers found")
                print(f"  [FRONTIER] Selected target at ({self.target[0]:.2f}, {self.target[1]:.2f}), "
                      f"distance={nearest.distance:.2f}m")

        # Navigate toward target
        target_heading = self.compute_heading_to_target(
            agent_x, agent_y, self.target[0], self.target[1]
        )

        # Compute heading error (signed, in range [-pi, pi])
        heading_error = target_heading - agent_yaw
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        if debug and self._frame_count % 30 == 0:
            print(f"  [FRONTIER] Target: ({self.target[0]:.2f}, {self.target[1]:.2f}), "
                  f"heading_error={math.degrees(heading_error):.1f}deg")

        # If facing target, move forward
        if abs(heading_error) < self.heading_tolerance:
            return FORWARD

        # Otherwise, rotate toward target
        if heading_error > 0:
            return ROTATE_LEFT
        else:
            return ROTATE_RIGHT

    def get_frontier_count(self) -> int:
        """Get number of current frontiers."""
        return len(self.frontiers)

    def reset(self):
        """Reset explorer state."""
        self.target = None
        self.target_cell = None
        self.frontiers = []
        self.exploration_complete = False
        self._frame_count = 0


def test_frontier_extraction():
    """Test frontier extraction on a simple grid."""
    # Create test grid: center is free, edges are unknown
    grid = np.full((20, 20), 127, dtype=np.uint8)  # All unknown

    # Mark center region as free
    grid[8:12, 8:12] = 255

    explorer = FrontierExplorer()
    frontiers = explorer.extract_frontiers(grid)

    print(f"Test grid: 20x20, center 4x4 is free")
    print(f"Found {len(frontiers)} frontier cells")
    print(f"Expected: perimeter of 4x4 region = ~12 cells")

    # Frontiers should be on the boundary of the free region
    assert len(frontiers) > 0, "Should find some frontiers"
    assert len(frontiers) <= 16, "Should find reasonable number of frontiers"
    print("Frontier extraction test passed!")


def test_frontier_selection():
    """Test nearest frontier selection."""
    # Create test grid
    grid = np.full((100, 100), 127, dtype=np.uint8)

    # Free region around agent (at center)
    grid[45:55, 45:55] = 255

    # Another free region far away (should not be selected)
    grid[80:90, 80:90] = 255

    def world_to_map(x, y):
        return int(x * 10 + 50), int(y * 10 + 50)

    def map_to_world(cx, cy):
        return (cx - 50) / 10, (cy - 50) / 10

    explorer = FrontierExplorer()
    frontiers = explorer.extract_frontiers(grid)

    agent_cell = (50, 50)  # Center
    nearest = explorer.select_nearest_frontier(frontiers, agent_cell, map_to_world)

    print(f"Agent at center, found {len(frontiers)} frontiers")
    if nearest:
        print(f"Nearest frontier at cell ({nearest.cell_x}, {nearest.cell_y}), "
              f"distance={nearest.distance:.2f}m")
    print("Frontier selection test passed!")


if __name__ == "__main__":
    test_frontier_extraction()
    print()
    test_frontier_selection()
