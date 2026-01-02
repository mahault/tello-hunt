"""
Comprehensive Tuning Test - Full Pipeline Integration.

Tests all frontier explorer tuning features with the complete escape handler:
1. Reduced escape scan frequency (6-11 vs 5-9)
2. Doorway detection and bonus scoring
3. Room transition tracking with door crossing events
4. Center-of-gap bias for narrow doorways
5. Blocked-edge memory for recovery

Runs for 500 frames with detailed statistics.

Run: python test_comprehensive_tuning.py
"""

import sys
import math
import numpy as np
import time
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, '.')

from simulator.glb_simulator import GLBSimulator
from pomdp.frontier_explorer import FrontierExplorer, RoomTransitionTracker
from utils.occupancy_map import OccupancyMap


@dataclass
class ExplorationStats:
    """Comprehensive exploration statistics."""
    frames: int = 0
    distance_traveled: float = 0.0
    unique_positions: int = 0
    rooms_visited: int = 0
    door_crossings: int = 0
    unique_doorways: int = 0
    escape_scans: int = 0
    collisions: int = 0
    targets_reached: int = 0
    targets_blacklisted: int = 0
    doorway_clusters_found: int = 0
    total_clusters_found: int = 0
    map_free_pct: float = 0.0
    map_occupied_pct: float = 0.0


class ComprehensiveExplorationTest:
    """
    Full exploration test with all tuning features.
    """

    def __init__(self, glb_path: str = "simulator/home_-_p3.glb"):
        print("=" * 70)
        print("COMPREHENSIVE EXPLORATION TEST")
        print("=" * 70)

        # Load simulator
        print("\n1. Loading simulator...")
        self.sim = GLBSimulator(glb_path=glb_path, width=640, height=480)

        # Initialize components
        print("\n2. Initializing components...")
        self.occ_map = OccupancyMap()
        self.explorer = FrontierExplorer()

        # Print tuning parameters
        print(f"   Escape thresholds: short=1-5, scan=6-11, clear=12+")
        print(f"   Doorway bonus: {self.explorer.doorway_bonus}m")
        print(f"   Doorway aspect min: {self.explorer.doorway_aspect_min}")

        # Coordinate transform
        self.origin_x, self.origin_z = -100, 333  # Kitchen center

        # Statistics
        self.stats = ExplorationStats()
        self._positions_visited = set()
        self._last_pos = None
        self._prev_frame = None

    def sim_to_world(self, sim_x: float, sim_z: float) -> Tuple[float, float]:
        """Convert simulator coords to world coords (meters)."""
        sx = (sim_x - self.origin_x) / 100.0
        sz = (sim_z - self.origin_z) / 100.0
        return -sz, sx

    def _scan_for_escape_direction(self) -> Tuple[float, int]:
        """
        Scan 360 degrees using actual sim.move() to find best escape direction.

        Returns (best_yaw, best_score) where score = number of successful steps.
        """
        # Snapshot ALL simulator state
        snapshot = self.sim.get_state_snapshot()

        best_yaw = snapshot['yaw']
        best_score = 0
        best_total_dist = 0.0

        # Get world coords for tie-breaking
        world_x, world_y = self.sim_to_world(snapshot['x'], snapshot['z'])

        # Get current frontier target for tie-breaking
        target_heading = None
        if self.explorer.target is not None:
            tx, ty = self.explorer.target
            target_heading = math.atan2(ty - world_y, tx - world_x)

        print(f"  [ESCAPE-SCAN] Starting 360° multi-step scan from ({snapshot['x']:.0f}, {snapshot['z']:.0f})")
        self.stats.escape_scans += 1

        # Try 16 directions (22.5 degree increments)
        candidates = []
        for angle_idx in range(16):
            test_yaw = angle_idx * (2 * math.pi / 16)

            # Restore to start position for this test
            self.sim.restore_state_snapshot(snapshot)
            self.sim.yaw = test_yaw

            # Try up to 5 forward steps, count successes
            steps_moved = 0
            total_dist = 0.0

            for step in range(5):
                old_x, old_z = self.sim.x, self.sim.z
                moved = self.sim.move(1, debug=False)  # FORWARD

                step_dist = math.sqrt((self.sim.x - old_x)**2 + (self.sim.z - old_z)**2)
                if step_dist > 1.0:  # Actually moved
                    steps_moved += 1
                    total_dist += step_dist
                else:
                    break

            # Score this direction
            score = steps_moved

            # Bonus for unknown cells (exploration value)
            if steps_moved > 0:
                # Check cells along path for unknown values
                check_x, check_y = self.sim_to_world(self.sim.x, self.sim.z)
                map_x, map_y = self.occ_map.world_to_map(check_x, check_y)
                if 0 <= map_x < self.occ_map.grid.shape[1] and 0 <= map_y < self.occ_map.grid.shape[0]:
                    cell_val = self.occ_map.grid[map_y, map_x]
                    if 100 < cell_val < 150:  # Unknown territory
                        score += 0.5

            # Bonus for alignment with target (if we have one)
            if target_heading is not None and steps_moved >= 2:
                heading_diff = abs(test_yaw - target_heading)
                if heading_diff > math.pi:
                    heading_diff = 2 * math.pi - heading_diff
                if heading_diff < math.pi / 4:  # Within 45 degrees
                    score += 0.3

            candidates.append((test_yaw, score, total_dist, steps_moved))

            if score > best_score or (score == best_score and total_dist > best_total_dist):
                best_score = score
                best_yaw = test_yaw
                best_total_dist = total_dist

        # Restore original state
        self.sim.restore_state_snapshot(snapshot)

        # Log top 3 candidates
        candidates.sort(key=lambda x: (-x[1], -x[2]))
        print(f"  [ESCAPE-SCAN] Top candidates:")
        for i, (yaw, score, dist, steps) in enumerate(candidates[:3]):
            print(f"    #{i+1}: yaw={math.degrees(yaw):.0f}° score={score:.1f} steps={steps} dist={dist:.0f}")

        return best_yaw, int(best_score)

    def _apply_gap_centering(self, world_x: float, world_y: float, world_yaw: float) -> float:
        """
        Apply center-of-gap bias when approaching narrow doorways.

        Returns adjusted yaw.
        """
        agent_cell = self.occ_map.world_to_map(world_x, world_y)

        correction = self.explorer.get_gap_steering_correction(
            grid=self.occ_map.grid,
            agent_cell=agent_cell,
            agent_yaw=world_yaw,
            world_to_map_fn=self.occ_map.world_to_map,
        )

        if abs(correction) > 0.01:  # Significant correction
            adjusted_yaw = world_yaw + correction
            print(f"  [GAP-CENTER] Applying {math.degrees(correction):.1f}° correction")
            return adjusted_yaw

        return world_yaw

    def run(self, max_frames: int = 500, visualize: bool = True) -> ExplorationStats:
        """
        Run the exploration test.

        Args:
            max_frames: Maximum frames to run
            visualize: Whether to show visualization

        Returns:
            ExplorationStats with comprehensive metrics
        """
        print(f"\n3. Running exploration ({max_frames} frames)...")
        print("   Controls: Q=quit, SPACE=pause, R=reset")

        # Set starting position
        self.sim.x, self.sim.z = -100, 333  # Kitchen center
        self.sim.yaw = 0.0
        self._last_pos = (self.sim.x, self.sim.z)

        paused = False
        frame = 0
        start_time = time.time()

        while frame < max_frames:
            # Handle visualization
            if visualize:
                # Render frame
                rgb = self.sim.render()
                depth_viz = self._render_depth_viz()
                map_viz = self.occ_map.render(size=300)

                # Combine displays
                rgb_small = cv2.resize(rgb, (400, 300))
                depth_small = cv2.resize(depth_viz, (400, 300))

                top_row = np.hstack([rgb_small, depth_small])
                map_padded = np.zeros((300, 800, 3), dtype=np.uint8)
                map_offset = (800 - 300) // 2
                map_padded[:, map_offset:map_offset+300] = map_viz

                # Add stats overlay
                stats_img = self._render_stats_overlay(frame, max_frames)
                map_padded[:100, :250] = stats_img

                display = np.vstack([top_row, map_padded])

                cv2.imshow("Comprehensive Exploration Test", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    print("\n   User quit")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"   {'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('r'):
                    self._reset()
                    frame = 0
                    continue

            if paused:
                time.sleep(0.1)
                continue

            # Get current state
            world_x, world_y = self.sim_to_world(self.sim.x, self.sim.z)
            world_yaw = self.sim.yaw

            # Update depth and occupancy
            depth_map = self.sim.get_depth_buffer()
            valid_depths = depth_map[(depth_map > 0) & (depth_map < 10000)]
            max_depth = max(np.percentile(valid_depths, 95), 100.0) if len(valid_depths) > 0 else 300.0
            rel_depth = 1.0 - np.clip(depth_map / max_depth, 0, 1)
            rel_depth = rel_depth.astype(np.float32)

            self.occ_map.update_from_depth(rel_depth, world_x, world_y, world_yaw, max_range=3.0)

            # Report room for transition tracking
            current_room = self.sim._get_current_room()
            transition = self.explorer.report_room(current_room, (world_x, world_y))
            if transition:
                self.stats.door_crossings += 1

            # === ESCAPE SCAN HANDLING ===
            if self.explorer.needs_escape_scan:
                best_yaw, best_score = self._scan_for_escape_direction()
                if best_score >= 2:
                    self.explorer.set_escape_direction(best_yaw)
                else:
                    print(f"  [ESCAPE-SCAN] No clear direction found (best score={best_score})")
                    self.explorer.cancel_escape()

            # Get action from explorer
            debug = (frame % 100 == 0)
            action = self.explorer.choose_action(
                grid=self.occ_map.grid,
                agent_x=world_x,
                agent_y=world_y,
                agent_yaw=world_yaw,
                world_to_map_fn=self.occ_map.world_to_map,
                map_to_world_fn=self.occ_map.map_to_world,
                debug=debug,
            )

            # Track cluster statistics
            for cluster in self.explorer.clusters:
                self.stats.total_clusters_found += 1
                if cluster.is_doorway:
                    self.stats.doorway_clusters_found += 1

            # Execute action
            if action > 0:
                old_x, old_z = self.sim.x, self.sim.z
                self.sim.move(action, debug=False)

                # Check if movement was blocked
                moved = abs(self.sim.x - old_x) > 5 or abs(self.sim.z - old_z) > 5

                if action == 1:  # FORWARD
                    if moved:
                        # Track distance
                        dist = math.sqrt((self.sim.x - old_x)**2 + (self.sim.z - old_z)**2)
                        self.stats.distance_traveled += dist / 100.0  # Convert to meters

                        # Report success to escape handler
                        self.explorer.report_escape_move_result(action, moved=True)
                    else:
                        # Collision handling
                        self.stats.collisions += 1
                        self.explorer.record_block(was_blocked=True)
                        self.occ_map.mark_obstacle_ahead(world_x, world_y, world_yaw)
                        self.explorer.report_escape_move_result(action, moved=False)

            # Track position
            pos_key = (int(self.sim.x / 20), int(self.sim.z / 20))
            self._positions_visited.add(pos_key)
            self.stats.unique_positions = len(self._positions_visited)

            # Progress report
            if frame % 100 == 0:
                room_stats = self.explorer.get_room_stats()
                elapsed = time.time() - start_time
                fps = frame / elapsed if elapsed > 0 else 0
                print(f"  Frame {frame}: room={current_room} "
                      f"pos=({self.sim.x:.0f},{self.sim.z:.0f}) "
                      f"doors={room_stats['total_transitions']} "
                      f"collisions={self.stats.collisions} "
                      f"fps={fps:.1f}")

            frame += 1
            self.stats.frames = frame

        # Final statistics
        self._finalize_stats()

        if visualize:
            cv2.destroyAllWindows()

        return self.stats

    def _render_depth_viz(self) -> np.ndarray:
        """Render depth buffer as grayscale visualization."""
        depth_map = self.sim.get_depth_buffer()
        valid_depths = depth_map[(depth_map > 0) & (depth_map < 10000)]
        if len(valid_depths) > 0:
            max_d = np.percentile(valid_depths, 95)
            depth_norm = np.clip(depth_map / max_d, 0, 1)
            depth_viz = (depth_norm * 255).astype(np.uint8)
        else:
            depth_viz = np.zeros(depth_map.shape, dtype=np.uint8)

        depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_GRAY2BGR)
        return depth_viz

    def _render_stats_overlay(self, frame: int, max_frames: int) -> np.ndarray:
        """Render statistics overlay."""
        img = np.zeros((100, 250, 3), dtype=np.uint8)

        room_stats = self.explorer.get_room_stats()

        lines = [
            f"Frame: {frame}/{max_frames}",
            f"Rooms: {room_stats['rooms_visited']} Doors: {room_stats['total_transitions']}",
            f"Collisions: {self.stats.collisions} Escapes: {self.stats.escape_scans}",
            f"Doorway clusters: {self.stats.doorway_clusters_found}",
        ]

        for i, line in enumerate(lines):
            cv2.putText(img, line, (5, 15 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return img

    def _finalize_stats(self):
        """Compute final statistics."""
        room_stats = self.explorer.get_room_stats()
        map_stats = self.occ_map.get_stats()

        self.stats.rooms_visited = room_stats['rooms_visited']
        self.stats.door_crossings = room_stats['total_transitions']
        self.stats.unique_doorways = room_stats['unique_doorways']
        self.stats.map_free_pct = map_stats['free_pct']
        self.stats.map_occupied_pct = map_stats['occupied_pct']

    def _reset(self):
        """Reset exploration state."""
        print("   Resetting...")
        self.sim.x, self.sim.z = -100, 333
        self.sim.yaw = 0.0
        self.occ_map.reset()
        self.explorer.reset()
        self.stats = ExplorationStats()
        self._positions_visited = set()
        self._last_pos = (self.sim.x, self.sim.z)

    def print_results(self):
        """Print comprehensive results."""
        print("\n" + "=" * 70)
        print("EXPLORATION RESULTS")
        print("=" * 70)

        room_stats = self.explorer.get_room_stats()

        print(f"""
Duration:
  Frames: {self.stats.frames}
  Distance traveled: {self.stats.distance_traveled:.1f}m
  Unique positions: {self.stats.unique_positions}

Room Exploration:
  Rooms visited: {room_stats['rooms_visited']}
  Room list: {room_stats['room_list']}
  Door crossings: {room_stats['total_transitions']}
  Unique doorways used: {room_stats['unique_doorways']}

Tuning Features:
  Doorway clusters detected: {self.stats.doorway_clusters_found}
  Total clusters analyzed: {self.stats.total_clusters_found}
  Doorway percentage: {100*self.stats.doorway_clusters_found/max(1,self.stats.total_clusters_found):.1f}%

Recovery:
  Collisions: {self.stats.collisions}
  Escape scans triggered: {self.stats.escape_scans}

Map Coverage:
  Free space: {self.stats.map_free_pct:.1f}%
  Occupied: {self.stats.map_occupied_pct:.1f}%
  Unknown: {100 - self.stats.map_free_pct - self.stats.map_occupied_pct:.1f}%

Recent Transitions:
  {room_stats['recent_transitions']}
""")


# Try to import cv2 for visualization
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available, running without visualization")


def run_test(max_frames: int = 300, visualize: bool = None):
    """Main test entry point."""
    test = ComprehensiveExplorationTest()

    if visualize is None:
        visualize = HAS_CV2

    try:
        stats = test.run(max_frames=max_frames, visualize=visualize)
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    test.print_results()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
