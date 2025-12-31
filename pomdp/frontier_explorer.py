# pomdp/frontier_explorer.py
"""
Frontier-Based Exploration for Tello Drone.

The canonical solution for exploring unknown connected environments:
1. Find frontiers (free cells adjacent to unknown)
2. Cluster frontiers into meaningful regions (doorways, corridor mouths)
3. Select nearest REACHABLE cluster centroid above minimum distance
4. Navigate toward it
5. Repeat until no frontiers remain

Three critical safeguards (missing these causes degenerate behavior):
1. Minimum frontier distance - reject sensor boundary noise
2. Clustering - treat doorways as single goals, not pixel soup
3. Reachability - only select frontiers connected via free space
"""

import numpy as np
import math
from typing import Tuple, List, Optional, Set
from dataclasses import dataclass
from collections import deque


@dataclass
class FrontierCluster:
    """A cluster of adjacent frontier cells."""
    cells: List[Tuple[int, int]]  # All cells in cluster
    centroid_cell: Tuple[int, int]  # Centroid in grid coords
    centroid_world: Tuple[float, float]  # Centroid in world coords
    size: int  # Number of cells
    distance: float  # Distance from agent to centroid


# Action indices matching the simulator
STAY = 0
FORWARD = 1
BACKWARD = 2
ROTATE_LEFT = 3
ROTATE_RIGHT = 4


class FrontierExplorer:
    """
    Frontier-based exploration with phased safeguards.

    EARLY phase (bootstrap): No filters - just pick nearest frontier
    MID phase (real exploration): Apply clustering, min distance, reachability

    This prevents logic deadlock at startup when explored area is tiny.
    """

    def __init__(
        self,
        min_frontier_distance: float = 0.5,   # SAFEGUARD 1: Reject micro-frontiers
        reached_threshold: float = 0.3,        # Consider target reached
        heading_tolerance: float = 0.35,       # ~20 deg - when to move forward
        min_cluster_size: int = 3,             # Ignore tiny clusters (noise)
        bootstrap_cells: int = 50,             # Cells needed before applying filters
    ):
        self.min_frontier_distance = min_frontier_distance
        self.reached_threshold = reached_threshold
        self.heading_tolerance = heading_tolerance
        self.min_cluster_size = min_cluster_size
        self.bootstrap_cells = bootstrap_cells

        # Current target
        self.target: Optional[Tuple[float, float]] = None
        self.target_cell: Optional[Tuple[int, int]] = None

        # State
        self.clusters: List[FrontierCluster] = []
        self.exploration_complete = False
        self._frame_count = 0
        self._no_valid_frontier_count = 0  # Track consecutive failures
        self._phase = "EARLY"  # EARLY or MID
        self._current_target_blocks = 0  # Blocks while pursuing CURRENT target
        self._blocked_targets: List[Tuple[Tuple[int, int], int]] = []  # (cell, expiry_frame)
        self._blacklist_duration = 50  # Frames to blacklist
        self._blacklist_radius = 3  # Cells - skip nearby clusters
        self._max_blacklisted = 8  # Max concurrent blacklisted targets
        self._blocks_to_blacklist = 2  # Blocks before blacklisting current target

    def extract_frontiers(
        self,
        grid: np.ndarray,
        unknown_val: int = 127,
        free_threshold: int = 180,
    ) -> List[Tuple[int, int]]:
        """
        Extract frontier cells from occupancy grid.
        A frontier is a FREE cell adjacent to an UNKNOWN cell.
        """
        h, w = grid.shape
        frontiers = []

        free_mask = grid > free_threshold
        unknown_mask = np.abs(grid.astype(np.int16) - unknown_val) < 20

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if free_mask[y, x]:
                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if unknown_mask[ny, nx]:
                            frontiers.append((x, y))
                            break

        return frontiers

    def cluster_frontiers(
        self,
        frontiers: List[Tuple[int, int]],
        map_to_world_fn,
        agent_world: Tuple[float, float],
    ) -> List[FrontierCluster]:
        """
        SAFEGUARD 2: Cluster adjacent frontier cells.

        Instead of treating each pixel as a goal, group them into
        meaningful regions (doorways, corridor mouths).
        """
        if not frontiers:
            return []

        frontier_set = set(frontiers)
        visited: Set[Tuple[int, int]] = set()
        clusters = []

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8-connectivity

        for start in frontiers:
            if start in visited:
                continue

            # BFS to find connected component
            cluster_cells = []
            queue = deque([start])
            visited.add(start)

            while queue:
                cell = queue.popleft()
                cluster_cells.append(cell)

                for dx, dy in neighbors:
                    neighbor = (cell[0] + dx, cell[1] + dy)
                    if neighbor in frontier_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # Skip tiny clusters (noise)
            if len(cluster_cells) < self.min_cluster_size:
                continue

            # Compute centroid
            cx = sum(c[0] for c in cluster_cells) // len(cluster_cells)
            cy = sum(c[1] for c in cluster_cells) // len(cluster_cells)

            wx, wy = map_to_world_fn(cx, cy)
            dist = math.sqrt((wx - agent_world[0])**2 + (wy - agent_world[1])**2)

            clusters.append(FrontierCluster(
                cells=cluster_cells,
                centroid_cell=(cx, cy),
                centroid_world=(wx, wy),
                size=len(cluster_cells),
                distance=dist,
            ))

        return clusters

    def is_reachable(
        self,
        grid: np.ndarray,
        start_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
        free_threshold: int = 180,
    ) -> bool:
        """
        SAFEGUARD 3: Check if goal is reachable via FREE space.

        Uses BFS flood-fill on free cells only.
        """
        h, w = grid.shape
        sx, sy = start_cell
        gx, gy = goal_cell

        # Bounds check
        if not (0 <= sx < w and 0 <= sy < h):
            return False
        if not (0 <= gx < w and 0 <= gy < h):
            return False

        # BFS
        visited = set()
        queue = deque([(sx, sy)])
        visited.add((sx, sy))

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()

            # Reached goal (or close enough - within 3 cells)
            if abs(x - gx) <= 3 and abs(y - gy) <= 3:
                return True

            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if (nx, ny) not in visited:
                        # Check if cell is free (traversable)
                        if grid[ny, nx] > free_threshold:
                            visited.add((nx, ny))
                            queue.append((nx, ny))

        return False

    def select_best_frontier(
        self,
        clusters: List[FrontierCluster],
        grid: np.ndarray,
        agent_cell: Tuple[int, int],
        agent_yaw: float = None,
        agent_world: Tuple[float, float] = None,
        debug: bool = False,
    ) -> Optional[FrontierCluster]:
        """
        Select the best frontier cluster:
        1. Must be above minimum distance (SAFEGUARD 1)
        2. Must be reachable via free space (SAFEGUARD 3)
        3. Must not be blacklisted (recently blocked)
        4. Must be IN FRONT of drone (no backward targets)
        5. Among valid clusters, pick nearest
        """
        valid = []

        for cluster in clusters:
            # SAFEGUARD 1: Minimum distance
            if cluster.distance < self.min_frontier_distance:
                continue

            # SAFEGUARD 3: Reachability
            if not self.is_reachable(grid, agent_cell, cluster.centroid_cell):
                continue

            # SAFEGUARD 4: Not blacklisted
            if self._is_blacklisted(cluster.centroid_cell):
                if debug:
                    print(f"  [FRONTIER] Skipping blacklisted cluster at {cluster.centroid_world}")
                continue

            # SAFEGUARD 5: Must be in front of drone (within 90°)
            if agent_yaw is not None and agent_world is not None:
                target_heading = math.atan2(
                    cluster.centroid_world[1] - agent_world[1],
                    cluster.centroid_world[0] - agent_world[0]
                )
                heading_diff = target_heading - agent_yaw
                while heading_diff > math.pi:
                    heading_diff -= 2 * math.pi
                while heading_diff < -math.pi:
                    heading_diff += 2 * math.pi

                if abs(heading_diff) > math.pi / 2:
                    if debug:
                        print(f"  [FRONTIER] Skipping backward cluster at {cluster.centroid_world}")
                    continue

            valid.append(cluster)

        if not valid:
            return None

        # Pick nearest valid cluster
        return min(valid, key=lambda c: c.distance)

    def _select_nearest_frontier_cell(
        self,
        frontiers: List[Tuple[int, int]],
        agent_cell: Tuple[int, int],
        map_to_world_fn,
        min_dist_cells: int = 0,  # Minimum distance in cells
        agent_yaw: float = None,  # Agent's current heading (if provided, prefer forward frontiers)
        agent_world: Tuple[float, float] = None,  # Agent's world position
    ) -> Optional[Tuple[Tuple[float, float], Tuple[int, int]]]:
        """
        Select nearest frontier cell, preferring frontiers in front of the drone.

        This prevents selecting targets behind the drone which would require
        180° rotation and appear as "going backward".
        """
        if not frontiers:
            return None

        ax, ay = agent_cell
        best_cell = None
        best_score = float('inf')

        for fx, fy in frontiers:
            dist = math.sqrt((fx - ax)**2 + (fy - ay)**2)
            # Skip frontiers too close (prevents selecting agent's own cell)
            if dist < min_dist_cells:
                continue
            # Skip blacklisted frontiers
            if self._is_blacklisted((fx, fy)):
                continue

            # If we know the agent's heading, REJECT targets behind us
            if agent_yaw is not None and agent_world is not None:
                world_pos = map_to_world_fn(fx, fy)
                target_heading = math.atan2(
                    world_pos[1] - agent_world[1],
                    world_pos[0] - agent_world[0]
                )
                heading_diff = target_heading - agent_yaw
                # Normalize to [-π, π]
                while heading_diff > math.pi:
                    heading_diff -= 2 * math.pi
                while heading_diff < -math.pi:
                    heading_diff += 2 * math.pi

                # REJECT targets that require more than 90° rotation
                # This prevents "going backward" - drone only moves where camera can see
                if abs(heading_diff) > math.pi / 2:
                    continue  # Skip this frontier entirely

            # Score is just distance (all remaining targets are in front)
            if dist < best_score:
                best_score = dist
                best_cell = (fx, fy)

        if best_cell is None:
            return None

        world_pos = map_to_world_fn(best_cell[0], best_cell[1])
        return (world_pos, best_cell)

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
        Choose action using frontier-based exploration with PHASED safeguards.

        EARLY phase: No filters, just pick nearest frontier cell
        MID phase: Apply clustering, min distance, reachability
        """
        self._frame_count += 1

        agent_cell = world_to_map_fn(agent_x, agent_y)
        agent_world = (agent_x, agent_y)

        # Count explored cells to determine phase
        free_threshold = 180
        explored_cells = np.sum(grid > free_threshold)
        old_phase = self._phase
        self._phase = "EARLY" if explored_cells < self.bootstrap_cells else "MID"

        if old_phase != self._phase and debug:
            print(f"  [FRONTIER] Phase transition: {old_phase} -> {self._phase} ({explored_cells} cells explored)")

        # Check if we've reached current target
        if self.target is not None:
            dist_to_target = math.sqrt(
                (agent_x - self.target[0])**2 +
                (agent_y - self.target[1])**2
            )
            if dist_to_target < self.reached_threshold:
                if debug:
                    print(f"  [FRONTIER] Reached target at ({self.target[0]:.2f}, {self.target[1]:.2f})")
                self.target = None
                self.target_cell = None

        # Check if current target is still valid
        if self.target_cell is not None:
            # Invalidate if target became an obstacle
            tx, ty = self.target_cell
            if 0 <= tx < grid.shape[1] and 0 <= ty < grid.shape[0]:
                cell_val = grid[ty, tx]
                if cell_val < 50:  # Definitely an obstacle
                    if debug:
                        print(f"  [FRONTIER] Target invalidated (obstacle)")
                    self.target = None
                    self.target_cell = None

            # Invalidate if target is now BEHIND the drone (more than 90°)
            # This prevents backward movement after rotations
            if self.target is not None:
                target_heading = math.atan2(
                    self.target[1] - agent_y,
                    self.target[0] - agent_x
                )
                heading_diff = target_heading - agent_yaw
                while heading_diff > math.pi:
                    heading_diff -= 2 * math.pi
                while heading_diff < -math.pi:
                    heading_diff += 2 * math.pi

                if abs(heading_diff) > math.pi / 2:
                    if debug:
                        print(f"  [FRONTIER] Target abandoned (now behind drone)")
                    self.target = None
                    self.target_cell = None

        # Find new target if needed
        if self.target is None:
            # Extract raw frontiers
            frontiers = self.extract_frontiers(grid)

            if not frontiers:
                # Check if we have any FREE cells
                has_free = explored_cells > 0
                if not has_free:
                    # No observations yet - move forward to start exploring
                    if debug:
                        print(f"  [FRONTIER] No observations yet - moving forward")
                    return FORWARD

                self.exploration_complete = True
                if debug:
                    print(f"  [FRONTIER] No frontiers - exploration complete!")
                return STAY

            # === PHASE-DEPENDENT TARGET SELECTION ===
            if self._phase == "EARLY":
                # BOOTSTRAP: Pick nearest FORWARD frontier cell
                result = self._select_nearest_frontier_cell(
                    frontiers, agent_cell, map_to_world_fn,
                    min_dist_cells=3,  # At least 3 cells away to avoid self-selection
                    agent_yaw=agent_yaw,  # Prefer frontiers in front of drone
                    agent_world=agent_world
                )
                if result:
                    self.target, self.target_cell = result
                    self._current_target_blocks = 0  # Reset for new target
                    if debug:
                        print(f"  [FRONTIER] EARLY: nearest frontier at ({self.target[0]:.2f}, {self.target[1]:.2f})")
                else:
                    # Fallback: move forward to create frontiers
                    if debug:
                        print(f"  [FRONTIER] EARLY: no frontiers, moving forward")
                    return FORWARD

            else:
                # MID PHASE: Apply all safeguards
                self.clusters = self.cluster_frontiers(frontiers, map_to_world_fn, agent_world)

                if debug:
                    print(f"  [FRONTIER] MID: {len(frontiers)} cells -> {len(self.clusters)} clusters")

                # Select best cluster (applies min distance, reachability, blacklist, forward-only)
                best = self.select_best_frontier(
                    self.clusters, grid, agent_cell,
                    agent_yaw=agent_yaw, agent_world=agent_world, debug=debug
                )

                if best is None:
                    # FALLBACK: Pick nearest FORWARD frontier
                    result = self._select_nearest_frontier_cell(
                        frontiers, agent_cell, map_to_world_fn,
                        min_dist_cells=3,
                        agent_yaw=agent_yaw,
                        agent_world=agent_world
                    )
                    if result:
                        self.target, self.target_cell = result
                        self._current_target_blocks = 0  # Reset for new target
                        if debug:
                            print(f"  [FRONTIER] MID fallback: nearest frontier at ({self.target[0]:.2f}, {self.target[1]:.2f})")
                    else:
                        # No frontiers in front - ROTATE to find them
                        # Don't move forward blindly as that could hit walls
                        if debug:
                            print(f"  [FRONTIER] No forward frontiers - rotating to scan")
                        return ROTATE_LEFT if self._frame_count % 2 == 0 else ROTATE_RIGHT
                else:
                    self._no_valid_frontier_count = 0
                    self.target = best.centroid_world
                    self.target_cell = best.centroid_cell
                    self._current_target_blocks = 0  # Reset for new target

                    if debug:
                        print(f"  [FRONTIER] MID: cluster {best.size} cells at "
                              f"({best.centroid_world[0]:.2f}, {best.centroid_world[1]:.2f}), "
                              f"dist={best.distance:.2f}m")

        # Navigate toward target
        target_heading = math.atan2(
            self.target[1] - agent_y,
            self.target[0] - agent_x
        )

        heading_error = target_heading - agent_yaw
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        if debug and self._frame_count % 30 == 0:
            dist = math.sqrt((agent_x - self.target[0])**2 + (agent_y - self.target[1])**2)
            print(f"  [FRONTIER] Navigating: dist={dist:.2f}m, heading_error={math.degrees(heading_error):.1f}deg")

        # If facing target, check if path looks clear before moving
        if abs(heading_error) < self.heading_tolerance:
            # Check MULTIPLE points ahead for obstacles
            check_distances = [0.2, 0.4, 0.6]  # meters
            for check_dist in check_distances:
                check_x = agent_x + check_dist * math.cos(agent_yaw)
                check_y = agent_y + check_dist * math.sin(agent_yaw)
                check_cell = world_to_map_fn(check_x, check_y)

                if (0 <= check_cell[0] < grid.shape[1] and
                    0 <= check_cell[1] < grid.shape[0]):
                    cell_val = grid[check_cell[1], check_cell[0]]
                    if cell_val < 50:  # Only block on definite obstacles (value < 50)
                        # Direction is blocked - blacklist this target and rotate
                        print(f"  [FRONTIER] Path blocked at {check_dist:.1f}m (cell={cell_val}), blacklisting")
                        if self.target_cell is not None:
                            expiry = self._frame_count + self._blacklist_duration
                            self._blocked_targets.append((self.target_cell, expiry))
                        self.target = None
                        self.target_cell = None
                        self._current_target_blocks = 0
                        # Rotate to explore a different direction
                        return ROTATE_LEFT if self._frame_count % 2 == 0 else ROTATE_RIGHT

            return FORWARD

        # Otherwise rotate toward target
        if heading_error > 0:
            return ROTATE_LEFT
        else:
            return ROTATE_RIGHT

    def get_cluster_count(self) -> int:
        return len(self.clusters)

    def record_block(self, was_blocked: bool):
        """
        Record that a movement was blocked while pursuing current target.

        Key insight: If we're blocked while near the target, blacklist immediately.
        This prevents getting stuck on targets that are technically "reachable"
        but blocked by walls.
        """
        if was_blocked and self.target_cell is not None:
            self._current_target_blocks += 1
            print(f"  [FRONTIER] Block #{self._current_target_blocks} for target {self.target}")

            # Blacklist immediately (was 2 blocks, now 1)
            # This prevents getting stuck on unreachable targets
            print(f"  [FRONTIER] Blacklisting target {self.target} after block")
            expiry = self._frame_count + self._blacklist_duration
            self._blocked_targets.append((self.target_cell, expiry))
            self.target = None
            self.target_cell = None
            self._current_target_blocks = 0  # Reset for next target

    def _is_blacklisted(self, cell: Tuple[int, int]) -> bool:
        """Check if a cell is currently blacklisted."""
        # Clean up expired entries
        self._blocked_targets = [
            (c, exp) for c, exp in self._blocked_targets
            if exp > self._frame_count
        ]
        # Limit max blacklisted targets (FIFO eviction)
        while len(self._blocked_targets) > self._max_blacklisted:
            self._blocked_targets.pop(0)

        # Check if cell is near any blacklisted target (use smaller radius)
        for blocked_cell, _ in self._blocked_targets:
            dist = math.sqrt((cell[0] - blocked_cell[0])**2 + (cell[1] - blocked_cell[1])**2)
            if dist < self._blacklist_radius:
                return True
        return False

    def reset(self):
        self.target = None
        self.target_cell = None
        self.clusters = []
        self.exploration_complete = False
        self._frame_count = 0
        self._no_valid_frontier_count = 0
        self._phase = "EARLY"
        self._current_target_blocks = 0
        self._blocked_targets = []


def test_clustering():
    """Test frontier clustering."""
    grid = np.full((100, 100), 127, dtype=np.uint8)

    # Create free region
    grid[40:60, 40:60] = 255

    # Create a "doorway" - narrow opening to unknown
    grid[50:55, 60:65] = 255  # Corridor extending right

    def map_to_world(cx, cy):
        return (cx - 50) / 10, (cy - 50) / 10

    explorer = FrontierExplorer(min_frontier_distance=0.3)
    frontiers = explorer.extract_frontiers(grid)
    clusters = explorer.cluster_frontiers(frontiers, map_to_world, (0, 0))

    print(f"Found {len(frontiers)} frontier cells")
    print(f"Clustered into {len(clusters)} groups:")
    for i, c in enumerate(clusters):
        print(f"  Cluster {i}: {c.size} cells, centroid={c.centroid_world}, dist={c.distance:.2f}m")

    print("Clustering test passed!")


def test_reachability():
    """Test reachability check."""
    grid = np.full((50, 50), 127, dtype=np.uint8)

    # Free corridor
    grid[20:30, 10:40] = 255

    # Wall blocking direct path
    grid[25, 25] = 0  # Obstacle

    explorer = FrontierExplorer()

    # Should be reachable (can go around)
    reachable1 = explorer.is_reachable(grid, (15, 25), (35, 25))
    print(f"Path around obstacle: {reachable1}")

    # Block completely
    grid[20:30, 25] = 0  # Full wall
    reachable2 = explorer.is_reachable(grid, (15, 25), (35, 25))
    print(f"Path through wall: {reachable2}")

    print("Reachability test passed!")


if __name__ == "__main__":
    test_clustering()
    print()
    test_reachability()
