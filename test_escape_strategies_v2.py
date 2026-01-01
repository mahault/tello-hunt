"""
Test escape strategies V2 - with realistic grid population using simulator raycast.

Key improvement: Instead of manually marking obstacles, we use the simulator's
raycast collision detection to populate the grid, matching real pipeline behavior.
"""

import sys
import math
import numpy as np
from typing import Tuple, List, Optional

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

    def choose_action(self, grid: np.ndarray, agent_x: float, agent_y: float,
                     agent_yaw: float, world_to_map_fn, map_to_world_fn,
                     check_collision_fn=None) -> int:
        raise NotImplementedError


class BackwardEscapeStrategy(EscapeStrategy):
    """Current: back up + cooldown."""

    def __init__(self):
        super().__init__("BACKWARD")
        self._no_forward_until_frame = 0

    def reset(self):
        super().reset()
        self._no_forward_until_frame = 0

    def choose_action(self, grid: np.ndarray, agent_x: float, agent_y: float,
                     agent_yaw: float, world_to_map_fn, map_to_world_fn,
                     check_collision_fn=None) -> int:
        self.frame_count += 1

        # Check if blocked
        blocked = check_collision_fn(agent_x, agent_y, agent_yaw, 0.5) if check_collision_fn else False

        if blocked:
            self.stuck_streak += 1

            if self.stuck_streak <= 6:
                return ROTATE_LEFT if (self.stuck_streak % 2 == 0) else ROTATE_RIGHT

            if self.stuck_streak == 7:
                self._no_forward_until_frame = self.frame_count + 15
                return BACKWARD

            return ROTATE_LEFT if (self.frame_count % 2 == 0) else ROTATE_RIGHT
        else:
            if self.frame_count < self._no_forward_until_frame:
                return ROTATE_LEFT if (self.frame_count % 2 == 0) else ROTATE_RIGHT

            self.stuck_streak = 0
            return FORWARD


class LocalFrontierEscapeStrategy(EscapeStrategy):
    """Local frontier: find nearby free cells, rotate toward best, move."""

    def __init__(self):
        super().__init__("LOCAL_FRONTIER")
        self._escape_goal_cell: Optional[Tuple[int, int]] = None
        self._escape_steps = 0

    def reset(self):
        super().reset()
        self._escape_goal_cell = None
        self._escape_steps = 0

    def _find_escape_goal(self, grid: np.ndarray, agent_cell: Tuple[int, int],
                         agent_yaw: float, check_collision_fn, agent_x: float, agent_y: float,
                         world_to_map_fn, map_to_world_fn) -> Optional[Tuple[int, int]]:
        """Find escape goal using both grid AND collision check."""
        ax, ay = agent_cell
        h, w = grid.shape

        best_cell = None
        best_score = -float('inf')

        # Try directions in 22.5-degree increments around full circle
        for angle_idx in range(16):
            angle = angle_idx * (2 * math.pi / 16)

            # Estimate clearance at this heading using collision check
            clearance = 0.0
            for dist in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
                if check_collision_fn:
                    if check_collision_fn(agent_x, agent_y, angle, dist):
                        break
                    clearance = dist
                else:
                    # Fallback to grid
                    check_x = agent_x + dist * math.cos(angle)
                    check_y = agent_y + dist * math.sin(angle)
                    cx, cy = world_to_map_fn(check_x, check_y)
                    if 0 <= cx < w and 0 <= cy < h:
                        if grid[cy, cx] < 50:
                            break
                    clearance = dist

            if clearance < 0.2:
                continue  # Skip blocked directions

            # Score this direction
            score = clearance * 2.0  # Base: how far we can go

            # Prefer forward-ish directions (less rotation needed)
            angle_diff = abs(angle - agent_yaw)
            while angle_diff > math.pi:
                angle_diff = abs(angle_diff - 2 * math.pi)
            score -= angle_diff * 0.5  # Penalize rotation

            # Slight novelty bonus (unknown cells are good for exploration)
            target_x = agent_x + clearance * math.cos(angle)
            target_y = agent_y + clearance * math.sin(angle)
            cx, cy = world_to_map_fn(target_x, target_y)
            if 0 <= cx < w and 0 <= cy < h:
                cell_val = grid[cy, cx]
                if 100 < cell_val < 180:  # Unknown-ish
                    score += 0.5

            if score > best_score:
                best_score = score
                # Store the target cell
                target_x = agent_x + clearance * 0.8 * math.cos(angle)
                target_y = agent_y + clearance * 0.8 * math.sin(angle)
                best_cell = world_to_map_fn(target_x, target_y)

        return best_cell

    def choose_action(self, grid: np.ndarray, agent_x: float, agent_y: float,
                     agent_yaw: float, world_to_map_fn, map_to_world_fn,
                     check_collision_fn=None) -> int:
        self.frame_count += 1
        agent_cell = world_to_map_fn(agent_x, agent_y)

        blocked = check_collision_fn(agent_x, agent_y, agent_yaw, 0.4) if check_collision_fn else False

        if blocked:
            self.stuck_streak += 1
        else:
            self.stuck_streak = max(0, self.stuck_streak - 1)

        # Enter escape mode if stuck
        if self.stuck_streak >= 4 and not self.escape_mode:
            self._escape_goal_cell = self._find_escape_goal(
                grid, agent_cell, agent_yaw, check_collision_fn,
                agent_x, agent_y, world_to_map_fn, map_to_world_fn
            )
            if self._escape_goal_cell is not None:
                self.escape_mode = True
                self._escape_steps = 0
                self.escape_until_frame = self.frame_count + 25

        # Execute escape
        if self.escape_mode:
            if self._escape_goal_cell is None or self.frame_count > self.escape_until_frame:
                self.escape_mode = False
                return ROTATE_LEFT

            goal_x, goal_y = map_to_world_fn(self._escape_goal_cell[0], self._escape_goal_cell[1])
            target_yaw = math.atan2(goal_y - agent_y, goal_x - agent_x)

            heading_error = target_yaw - agent_yaw
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi

            dist_to_goal = math.sqrt((goal_x - agent_x)**2 + (goal_y - agent_y)**2)
            if dist_to_goal < 0.15:
                self.escape_mode = False
                self.stuck_streak = 0
                return FORWARD

            if abs(heading_error) > 0.25:
                return ROTATE_LEFT if heading_error > 0 else ROTATE_RIGHT

            # Check if forward is now clear
            if check_collision_fn and check_collision_fn(agent_x, agent_y, agent_yaw, 0.3):
                # Still blocked in this direction - abort and try again
                self.escape_mode = False
                return ROTATE_LEFT

            self._escape_steps += 1
            if self._escape_steps >= 6:
                self.escape_mode = False
                self.stuck_streak = 0
            return FORWARD

        if not blocked:
            return FORWARD
        else:
            return ROTATE_LEFT if (self.frame_count % 2 == 0) else ROTATE_RIGHT


class RotateAndGoEscapeStrategy(EscapeStrategy):
    """Rotate-and-go: scan headings, pick best clearance, rotate there, move."""

    def __init__(self):
        super().__init__("ROTATE_AND_GO")
        self._target_heading: Optional[float] = None
        self._escape_steps = 0

    def reset(self):
        super().reset()
        self._target_heading = None
        self._escape_steps = 0

    def _find_best_heading(self, agent_x: float, agent_y: float, agent_yaw: float,
                          check_collision_fn) -> Tuple[float, float]:
        """Scan full 360 degrees for best clearance."""
        best_heading = agent_yaw
        best_clearance = 0.0

        # Try 16 directions (22.5 deg increments) around full circle
        for angle_idx in range(16):
            test_heading = angle_idx * (2 * math.pi / 16)

            clearance = 0.0
            for dist in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
                if check_collision_fn and check_collision_fn(agent_x, agent_y, test_heading, dist):
                    break
                clearance = dist

            # Small preference for directions requiring less rotation
            angle_diff = abs(test_heading - agent_yaw)
            while angle_diff > math.pi:
                angle_diff = abs(angle_diff - 2 * math.pi)

            effective_clearance = clearance - angle_diff * 0.05

            if effective_clearance > best_clearance:
                best_clearance = clearance
                best_heading = test_heading

        return best_heading, best_clearance

    def choose_action(self, grid: np.ndarray, agent_x: float, agent_y: float,
                     agent_yaw: float, world_to_map_fn, map_to_world_fn,
                     check_collision_fn=None) -> int:
        self.frame_count += 1

        blocked = check_collision_fn(agent_x, agent_y, agent_yaw, 0.4) if check_collision_fn else False

        if blocked:
            self.stuck_streak += 1
        else:
            self.stuck_streak = max(0, self.stuck_streak - 1)

        # Enter escape mode if stuck
        if self.stuck_streak >= 4 and not self.escape_mode:
            best_heading, clearance = self._find_best_heading(
                agent_x, agent_y, agent_yaw, check_collision_fn
            )

            if clearance > 0.3:
                self._target_heading = best_heading
                self.escape_mode = True
                self._escape_steps = 0
                self.escape_until_frame = self.frame_count + 25
            else:
                return ROTATE_LEFT

        # Execute escape
        if self.escape_mode:
            if self._target_heading is None or self.frame_count > self.escape_until_frame:
                self.escape_mode = False
                return ROTATE_LEFT

            heading_error = self._target_heading - agent_yaw
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi

            if abs(heading_error) > 0.2:
                return ROTATE_LEFT if heading_error > 0 else ROTATE_RIGHT

            # Check if forward is clear before moving
            if check_collision_fn and check_collision_fn(agent_x, agent_y, agent_yaw, 0.3):
                self.escape_mode = False
                return ROTATE_LEFT

            self._escape_steps += 1
            if self._escape_steps >= 6:
                self.escape_mode = False
                self.stuck_streak = 0
            return FORWARD

        if not blocked:
            return FORWARD
        else:
            return ROTATE_LEFT if (self.frame_count % 2 == 0) else ROTATE_RIGHT


def run_test(strategy: EscapeStrategy, sim: GLBSimulator, occ_map: OccupancyMap,
            corner_pos: Tuple[float, float, float], max_frames: int = 80) -> dict:
    """Run escape test with realistic collision checking."""

    strategy.reset()
    sim.x, sim.z = corner_pos[0], corner_pos[1]
    sim.yaw = math.radians(corner_pos[2])

    start_x = sim.model_center[0] if hasattr(sim, 'model_center') else -102
    start_z = sim.model_center[2] if hasattr(sim, 'model_center') else 333

    def sim_to_world(sim_x, sim_z):
        return -(sim_z - start_z) / 165.0, -(sim_x - start_x) / 165.0

    def world_to_sim(world_x, world_y):
        sim_z = start_z - world_x * 165.0
        sim_x = start_x - world_y * 165.0
        return sim_x, sim_z

    def check_collision(agent_x, agent_y, heading, dist):
        """Use simulator raycast to check collision."""
        # Convert world coords back to sim coords for raycast
        sim_x_curr, sim_z_curr = world_to_sim(agent_x, agent_y)

        # Check point ahead in world coords, then convert
        check_x = agent_x + dist * math.cos(heading)
        check_y = agent_y + dist * math.sin(heading)
        sim_x_target, sim_z_target = world_to_sim(check_x, check_y)

        # Use simulator's collision check
        if hasattr(sim, 'ray_tracer') and sim.ray_tracer is not None:
            # Raycast from current position toward target
            origin = np.array([sim_x_curr, sim.y, sim_z_curr])
            direction = np.array([sim_x_target - sim_x_curr, 0, sim_z_target - sim_z_curr])
            direction_len = np.linalg.norm(direction)
            if direction_len > 0.01:
                direction = direction / direction_len
                # Check if any hit within distance
                hit = sim.ray_tracer.ray_intersects(origin, direction)
                if hit is not None:
                    hit_dist = np.linalg.norm(hit - origin)
                    if hit_dist < dist * 165.0 + 10:  # Convert world dist to sim units
                        return True
        return False

    positions = []
    actions_taken = []
    initial_pos = sim_to_world(sim.x, sim.z)

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
            check_collision_fn=check_collision,
        )

        old_x, old_z = sim.x, sim.z
        if action > 0:
            sim.move(action, debug=False)

        positions.append((sim.x, sim.z))
        actions_taken.append(ACTION_NAMES[action])

    final_pos = sim_to_world(sim.x, sim.z)
    total_distance = math.sqrt((final_pos[0] - initial_pos[0])**2 + (final_pos[1] - initial_pos[1])**2)
    unique_positions = len(set((int(p[0]/10), int(p[1]/10)) for p in positions))
    escaped = total_distance > 0.3

    # Count stuck frames (forward issued but position didn't change)
    stuck = 0
    for i, a in enumerate(actions_taken):
        if a == 'FORWARD' and i > 0:
            if abs(positions[i][0] - positions[i-1][0]) < 0.5 and abs(positions[i][1] - positions[i-1][1]) < 0.5:
                stuck += 1

    return {
        'strategy': strategy.name,
        'total_distance': total_distance,
        'unique_positions': unique_positions,
        'stuck_frames': stuck,
        'escaped': escaped,
        'action_counts': {a: actions_taken.count(a) for a in set(actions_taken)},
        'final_pos': (sim.x, sim.z),
    }


def main():
    print("=" * 70)
    print("ESCAPE STRATEGY COMPARISON V2 (with raycast collision)")
    print("=" * 70)

    print("\n1. Loading simulator...")
    sim = GLBSimulator(glb_path="simulator/home_-_p3.glb", width=640, height=480)

    print("2. Initializing occupancy map...")
    occ_map = OccupancyMap()

    # Test corners in different rooms
    corners = [
        (-298.0, 280.0, 210, "Living Room corner"),
        (-380.0, 440.0, 45, "Living Room back corner"),
        (-10.0, 440.0, 135, "Kitchen corner"),
    ]

    strategies = [
        BackwardEscapeStrategy(),
        LocalFrontierEscapeStrategy(),
        RotateAndGoEscapeStrategy(),
    ]

    all_results = []

    for corner in corners:
        pos = corner[:3]
        name = corner[3]

        print(f"\n{'='*70}")
        print(f"CORNER: {name}")
        print(f"Position: sim=({pos[0]:.0f}, {pos[1]:.0f}) yaw={pos[2]}deg")
        print("-" * 70)

        for strategy in strategies:
            result = run_test(strategy, sim, occ_map, pos, max_frames=80)
            result['corner'] = name
            all_results.append(result)

            esc_mark = "YES" if result['escaped'] else "no"
            print(f"  {strategy.name:<18}: dist={result['total_distance']:.3f}m  "
                  f"unique={result['unique_positions']:2}  stuck={result['stuck_frames']:2}  "
                  f"escaped={esc_mark}")

    # Aggregate scores
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    strategy_scores = {}
    for s in strategies:
        results = [r for r in all_results if r['strategy'] == s.name]
        total_dist = sum(r['total_distance'] for r in results)
        total_escaped = sum(1 for r in results if r['escaped'])
        total_stuck = sum(r['stuck_frames'] for r in results)

        score = total_escaped * 5.0 + total_dist * 2.0 - total_stuck * 0.1
        strategy_scores[s.name] = {
            'total_dist': total_dist,
            'escapes': total_escaped,
            'stuck': total_stuck,
            'score': score
        }

    print(f"\n{'Strategy':<18} {'Total Dist':>12} {'Escapes':>10} {'Stuck':>8} {'Score':>10}")
    print("-" * 60)
    for name, data in sorted(strategy_scores.items(), key=lambda x: -x[1]['score']):
        print(f"{name:<18} {data['total_dist']:>12.3f}m {data['escapes']:>10}/3 {data['stuck']:>8} {data['score']:>10.2f}")

    best = max(strategy_scores.items(), key=lambda x: x[1]['score'])
    print(f"\nBEST OVERALL: {best[0]} (score: {best[1]['score']:.2f})")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
