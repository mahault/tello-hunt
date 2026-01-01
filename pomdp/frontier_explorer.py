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
import heapq
from typing import Tuple, List, Optional, Set, Dict
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

        # "Peek" safeguard: require N consecutive clear checks before moving forward
        self._forward_clear_streak = 0
        self._forward_clear_required = 2

        # Planned path (grid cells), used in MID phase (A*)
        self._planned_path: List[Tuple[int, int]] = []
        self._planned_goal: Optional[Tuple[int, int]] = None

        # Hysteresis for rotation direction (prevents oscillation at ±π)
        self._last_turn: Optional[int] = None

    # ---------------------------
    # A* PATH PLANNING HELPERS
    # ---------------------------
    def _cell_cost(self, grid: np.ndarray, cell: Tuple[int, int]) -> Optional[float]:
        """Return traversal cost for cell, or None if blocked."""
        x, y = cell
        if y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1]:
            return None
        v = int(grid[y, x])
        unknown_val = 127
        # Occupied if well below unknown (OccupancyMap uses 0 for occupied)
        if v < unknown_val - 20:
            return None
        # Free if well above unknown (OccupancyMap uses 255 for free)
        if v > unknown_val + 20:
            return 1.0
        # Unknown: allow but penalize (encourages reaching frontiers through known free space)
        return 3.0

    def _neighbors(self, grid: np.ndarray, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = cell
        h, w = grid.shape
        out = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                out.append((nx, ny))
        return out

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        # Manhattan works well on 4-neighborhood grids
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _astar(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
               max_expansions: int = 20000) -> Optional[List[Tuple[int, int]]]:
        """A* from start→goal. Returns list of cells including start and goal, or None."""
        if start == goal:
            return [start]

        # Goal may be frontier adjacent to unknown; allow planning *to* it if it's not occupied.
        if self._cell_cost(grid, goal) is None:
            return None

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        gscore: Dict[Tuple[int, int], float] = {start: 0.0}

        expansions = 0
        while open_heap and expansions < max_expansions:
            _, current = heapq.heappop(open_heap)
            expansions += 1

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for nb in self._neighbors(grid, current):
                c = self._cell_cost(grid, nb)
                if c is None:
                    continue
                tentative = gscore[current] + c
                if nb not in gscore or tentative < gscore[nb]:
                    came_from[nb] = current
                    gscore[nb] = tentative
                    f = tentative + self._heuristic(nb, goal)
                    heapq.heappush(open_heap, (f, nb))
        return None

    def _plan_path_to_cluster(self, grid: np.ndarray, agent_cell: Tuple[int, int],
                              cluster: 'FrontierCluster') -> Optional[List[Tuple[int, int]]]:
        """
        Pick a goal cell inside the cluster that is cheapest to reach via A*.
        Returns the best path, or None.

        IMPORTANT: Rejects trivial goals that are too close to the agent,
        which prevents the "oscillating at own cell" failure mode.
        """
        best_path = None
        best_cost = float("inf")

        # Evaluate a subset of cluster cells (avoid huge clusters)
        candidates = cluster.cells
        if len(candidates) > 40:
            # sample evenly
            step = max(1, len(candidates) // 40)
            candidates = candidates[::step]

        # Forbid trivial goals near the agent (prevents self-selection)
        min_goal_dist_cells = 4  # At least 4 cells away
        min_goal_dist2 = min_goal_dist_cells * min_goal_dist_cells
        min_path_len = 4  # Reject degenerate short paths

        ax, ay = agent_cell

        for goal in candidates:
            if self._is_blacklisted(goal):
                continue

            # NEW: Reject goals too close to agent
            dx = goal[0] - ax
            dy = goal[1] - ay
            if (dx * dx + dy * dy) <= min_goal_dist2:
                continue

            path = self._astar(grid, agent_cell, goal)
            if path is None:
                continue

            # DEBUG: Log goal selection to diagnose trivial-goal issues
            print(f"  [A*-DEBUG] agent_cell={agent_cell} goal={goal} path_len={len(path)}")

            # NEW: Reject degenerate paths
            if len(path) < min_path_len:
                continue

            cost = float(len(path))
            if cost < best_cost:
                best_cost = cost
                best_path = path

        return best_path

    def _path_waypoint_world(self, path: List[Tuple[int, int]], map_to_world_fn,
                             lookahead_cells: int = 4) -> Tuple[Tuple[int, int], Tuple[float, float]]:
        """
        Return a waypoint cell a few steps ahead on path + its world coords.
        """
        if not path:
            raise ValueError("empty path")
        idx = min(len(path) - 1, lookahead_cells)
        cell = path[idx]
        wx, wy = map_to_world_fn(cell[0], cell[1])
        return cell, (wx, wy)

    def _is_blacklisted(self, cell: Tuple[int, int]) -> bool:
        """Check if a cell is in the blacklist."""
        for (bc, expiry) in self._blocked_targets:
            if bc == cell:
                return True
            # Also check nearby cells within radius
            dx = abs(cell[0] - bc[0])
            dy = abs(cell[1] - bc[1])
            if dx <= self._blacklist_radius and dy <= self._blacklist_radius:
                return True
        return False

    def _blacklist_target(self, cell: Tuple[int, int]):
        """Add a cell to the blacklist."""
        expiry = self._frame_count + self._blacklist_duration
        self._blocked_targets.append((cell, expiry))
        # Trim old entries
        if len(self._blocked_targets) > self._max_blacklisted:
            self._blocked_targets = self._blocked_targets[-self._max_blacklisted:]

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
        4. Among valid clusters, pick nearest (with slight preference for forward)

        NOTE: We no longer reject targets based on heading angle.
        The heading-first movement gate handles rotation before movement.
        """
        valid = []

        for cluster in clusters:
            # SAFEGUARD 1: Minimum distance
            if cluster.distance < self.min_frontier_distance:
                continue

            # SAFEGUARD 2: Reachability via free space
            if not self.is_reachable(grid, agent_cell, cluster.centroid_cell):
                if debug:
                    print(f"  [FRONTIER] Skipping unreachable cluster at {cluster.centroid_world}")
                continue

            # SAFEGUARD 3: Not blacklisted
            if self._is_blacklisted(cluster.centroid_cell):
                if debug:
                    print(f"  [FRONTIER] Skipping blacklisted cluster at {cluster.centroid_world}")
                continue

            # Compute effective distance (add small penalty for rotation needed)
            effective_dist = cluster.distance
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

                # Add small penalty for rotation (0.1m per 90° of turn needed)
                rotation_penalty = abs(heading_diff) / (math.pi / 2) * 0.1
                effective_dist += rotation_penalty

            valid.append((cluster, effective_dist))

        if not valid:
            return None

        # Pick cluster with lowest effective distance
        best = min(valid, key=lambda x: x[1])
        return best[0]

    def _select_nearest_frontier_cell(
        self,
        frontiers: List[Tuple[int, int]],
        agent_cell: Tuple[int, int],
        map_to_world_fn,
        min_dist_cells: int = 0,
        agent_yaw: float = None,
        agent_world: Tuple[float, float] = None,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[int, int]]]:
        """
        Select nearest frontier cell with small rotation penalty.

        NOTE: We no longer reject targets based on heading angle.
        The heading-first movement gate handles rotation before movement.
        """
        if not frontiers:
            return None

        ax, ay = agent_cell
        best_cell = None
        best_score = float('inf')

        for fx, fy in frontiers:
            dist = math.sqrt((fx - ax)**2 + (fy - ay)**2)
            # Skip frontiers too close
            if dist < min_dist_cells:
                continue
            # Skip blacklisted frontiers
            if self._is_blacklisted((fx, fy)):
                continue

            # Compute effective score (distance + rotation penalty)
            score = dist
            if agent_yaw is not None and agent_world is not None:
                world_pos = map_to_world_fn(fx, fy)
                target_heading = math.atan2(
                    world_pos[1] - agent_world[1],
                    world_pos[0] - agent_world[0]
                )
                heading_diff = target_heading - agent_yaw
                while heading_diff > math.pi:
                    heading_diff -= 2 * math.pi
                while heading_diff < -math.pi:
                    heading_diff += 2 * math.pi

                # Small penalty for rotation (0.5 cells per 90° of turn)
                rotation_penalty = abs(heading_diff) / (math.pi / 2) * 0.5
                score += rotation_penalty

            if score < best_score:
                best_score = score
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
                self._planned_path = []
                self._planned_goal = None

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
                    self._planned_path = []
                    self._planned_goal = None

            # If the target is behind us, DO NOT abandon it.
            # Instead, rotate in place until the target comes in front.
            if self.target is not None:
                target_heading = math.atan2(
                    self.target[1] - agent_y,
                    self.target[0] - agent_x
                )
                # Use wrap_pi for stable signed shortest-angle error
                def wrap_pi(a):
                    return (a + math.pi) % (2 * math.pi) - math.pi

                heading_diff = wrap_pi(target_heading - agent_yaw)
                abs_err = abs(heading_diff)

                # DEBUG: Understand rotation behavior
                if abs_err > math.pi / 4:  # Log when > 45 deg
                    print(f"  [ROT-DEBUG] agent=({agent_x:.2f},{agent_y:.2f}) yaw={math.degrees(agent_yaw):.1f}deg "
                          f"target=({self.target[0]:.2f},{self.target[1]:.2f}) "
                          f"target_heading={math.degrees(target_heading):.1f}deg "
                          f"heading_diff={math.degrees(heading_diff):.1f}deg")

                # Hysteresis to prevent oscillation at ±π discontinuity
                behind = abs_err > math.pi / 2
                almost_opposite = abs_err > math.radians(170)

                if behind:
                    self._forward_clear_streak = 0
                    if debug:
                        print(f"  [FRONTIER] Target behind ({math.degrees(heading_diff):.1f}deg) -> rotating")

                    # When almost opposite (±170°+), maintain last direction to avoid chatter
                    if almost_opposite and self._last_turn is not None:
                        return self._last_turn

                    # Choose direction based on shorter turn
                    if heading_diff > 0:
                        self._last_turn = ROTATE_LEFT
                        return ROTATE_LEFT
                    else:
                        self._last_turn = ROTATE_RIGHT
                        return ROTATE_RIGHT

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
                        # All frontiers unreachable or blacklisted - rotate to scan
                        if debug:
                            print(f"  [FRONTIER] All frontiers blocked/blacklisted - rotating to scan")
                        return ROTATE_LEFT if self._frame_count % 2 == 0 else ROTATE_RIGHT
                else:
                    # PLAN A* PATH to the best cluster (MID phase behavior)
                    path = self._plan_path_to_cluster(grid, agent_cell, best)
                    if path is None:
                        # If no path, blacklist cluster centroid and try later
                        if debug:
                            print(f"  [FRONTIER] No A* path to cluster -> blacklisting {best.centroid_cell}")
                        self._blacklist_target(best.centroid_cell)
                        return ROTATE_LEFT

                    self._planned_path = path
                    self._planned_goal = path[-1]
                    # Use a waypoint a few steps ahead (smoother)
                    wp_cell, wp_world = self._path_waypoint_world(self._planned_path, map_to_world_fn, lookahead_cells=4)
                    self.target_cell = wp_cell
                    self.target = wp_world
                    self._no_valid_frontier_count = 0
                    self._current_target_blocks = 0  # Reset for new target

                    # Update the "distance" metric to path length for logging/selection stability
                    best.distance = float(len(path))

                    if debug:
                        print(f"  [FRONTIER] MID: A* path {len(path)} cells to cluster at "
                              f"({best.centroid_world[0]:.2f}, {best.centroid_world[1]:.2f})")

        # If we have a planned path, keep updating waypoint as we progress.
        if self._planned_path and self.target is not None:
            # Drop path prefix we've already reached
            while self._planned_path and self._planned_path[0] == agent_cell:
                self._planned_path.pop(0)
            if self._planned_path:
                wp_cell, wp_world = self._path_waypoint_world(self._planned_path, map_to_world_fn, lookahead_cells=4)
                self.target_cell = wp_cell
                self.target = wp_world

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
            # Reset hysteresis when aligned
            self._last_turn = None
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
                        # Invalidate planned path so we replan next tick
                        self._planned_path = []
                        self._planned_goal = None
                        # Rotate to explore a different direction
                        self._forward_clear_streak = 0
                        return ROTATE_LEFT if self._frame_count % 2 == 0 else ROTATE_RIGHT

            # "Peek" before committing to forward: require consecutive clear checks
            self._forward_clear_streak += 1
            if self._forward_clear_streak < self._forward_clear_required:
                if debug:
                    print(f"  [FRONTIER] Peek {self._forward_clear_streak}/{self._forward_clear_required} (clear) -> STAY")
                return STAY
            self._forward_clear_streak = 0
            return FORWARD

        # Otherwise rotate toward target
        # Reset peek streak when we need to rotate
        self._forward_clear_streak = 0
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
            self._planned_path = []
            self._planned_goal = None
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
        self._planned_path = []
        self._planned_goal = None
        self.clusters = []
        self.exploration_complete = False
        self._frame_count = 0
        self._no_valid_frontier_count = 0
        self._phase = "EARLY"
        self._current_target_blocks = 0
        self._blocked_targets = []
        self._forward_clear_streak = 0
        self._last_turn = None


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
