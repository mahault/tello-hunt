"""
Obstacle Origin Diagnostic

Track where obstacle cells come from:
1. Depth sensing (update_from_depth with close readings)
2. Collision stamping (mark_obstacle_ahead when simulator blocks)

This helps identify if the doorway blockage is real or a false positive.
"""

import sys
import math
import numpy as np

sys.path.insert(0, '.')

from simulator.glb_simulator import GLBSimulator
from utils.occupancy_map import OccupancyMap, MapConfig


class TrackedOccupancyMap(OccupancyMap):
    """OccupancyMap that tracks source of obstacle cells."""

    def __init__(self, config=None):
        super().__init__(config)
        # Track origin of obstacle cells: 'depth' or 'collision'
        self.obstacle_origin = np.full(
            (self.config.height, self.config.width),
            '',
            dtype=object
        )
        self._pending_collision = False
        self._collision_pose = None

    def mark_next_as_collision(self, pose_x, pose_y, pose_yaw):
        """Mark next obstacle update as coming from collision detection."""
        self._pending_collision = True
        self._collision_pose = (pose_x, pose_y, pose_yaw)

    def _update_cell(self, x: int, y: int, free: bool):
        """Override to track obstacle origin."""
        if not self.is_valid_cell(x, y):
            return

        old_val = int(self.grid[y, x])

        # Call parent
        super()._update_cell(x, y, free)

        new_val = int(self.grid[y, x])

        # Track if this became an obstacle
        if new_val < 64 and old_val >= 64:  # Just became obstacle
            if self._pending_collision:
                self.obstacle_origin[y, x] = 'collision'
            else:
                self.obstacle_origin[y, x] = 'depth'

    def mark_obstacle_ahead(self, pose_x, pose_y, pose_yaw, distance=0.3):
        """Override to track collision-based obstacles."""
        self.mark_next_as_collision(pose_x, pose_y, pose_yaw)
        super().mark_obstacle_ahead(pose_x, pose_y, pose_yaw, distance)
        self._pending_collision = False

    def get_obstacle_stats(self):
        """Get statistics about obstacle origins."""
        depth_count = np.sum(self.obstacle_origin == 'depth')
        collision_count = np.sum(self.obstacle_origin == 'collision')
        total_obstacles = np.sum(self.grid < 64)
        return {
            'total': total_obstacles,
            'from_depth': depth_count,
            'from_collision': collision_count,
        }

    def print_obstacle_band(self, world_x, world_y, world_yaw, fwd_dist=0.6):
        """Print obstacles in a band ahead, showing their origins."""
        n_samples = 11
        lateral_range = 1.0

        print(f"\n{'Lateral':>8} {'MapX':>6} {'MapY':>6} {'Value':>6} {'Origin':>10}")
        print("-" * 50)

        for i in range(n_samples):
            lateral = -0.5 + (i / (n_samples - 1)) * lateral_range

            # Calculate world position
            fwd_x = math.cos(world_yaw)
            fwd_y = math.sin(world_yaw)
            right_x = math.cos(world_yaw - math.pi/2)
            right_y = math.sin(world_yaw - math.pi/2)

            sample_x = world_x + fwd_dist * fwd_x + lateral * right_x
            sample_y = world_y + fwd_dist * fwd_y + lateral * right_y

            map_x, map_y = self.world_to_map(sample_x, sample_y)

            if 0 <= map_x < self.grid.shape[1] and 0 <= map_y < self.grid.shape[0]:
                value = self.grid[map_y, map_x]
                origin = self.obstacle_origin[map_y, map_x]
                if value >= 64:
                    origin = '-'
            else:
                value = -1
                origin = 'OOB'

            print(f"{lateral:>8.2f} {map_x:>6} {map_y:>6} {value:>6} {origin:>10}")


def run_test():
    print("=" * 70)
    print("OBSTACLE ORIGIN DIAGNOSTIC")
    print("=" * 70)

    print("\n1. Loading simulator...")
    sim = GLBSimulator(glb_path="simulator/home_-_p3.glb", width=640, height=480)

    print("2. Creating tracked occupancy map...")
    occ_map = TrackedOccupancyMap()

    # Set origin for coord transform
    origin_x, origin_z = sim.x, sim.z

    def sim_to_world(sim_x, sim_z):
        sx = (sim_x - origin_x) / 100.0
        sz = (sim_z - origin_z) / 100.0
        return -sz, sx

    print("\n3. Running exploration with obstacle tracking (100 frames)...")

    # Start in a position where we can explore
    sim.x, sim.z = -200.0, 350.0  # More central starting point
    sim.yaw = 0.0

    stuck_streak = 0
    last_blocked_pos = None

    for frame in range(100):
        world_x, world_y = sim_to_world(sim.x, sim.z)
        world_yaw = sim.yaw

        # Get depth
        depth_map = sim.get_depth_buffer()
        valid_depths = depth_map[(depth_map > 0) & (depth_map < 10000)]
        max_depth = max(np.percentile(valid_depths, 95), 100.0) if len(valid_depths) > 0 else 300.0
        rel_depth = 1.0 - np.clip(depth_map / max_depth, 0, 1)
        rel_depth = rel_depth.astype(np.float32)

        # Update map from depth (obstacle origin = 'depth')
        occ_map.update_from_depth(rel_depth, world_x, world_y, world_yaw, max_range=3.0)

        # Try to move forward
        old_x, old_z = sim.x, sim.z
        moved = sim.move(1, debug=False)  # FORWARD

        if not moved or (abs(sim.x - old_x) < 1 and abs(sim.z - old_z) < 1):
            # Blocked - stamp obstacle (origin = 'collision')
            occ_map.mark_obstacle_ahead(world_x, world_y, world_yaw)
            stuck_streak += 1

            if last_blocked_pos != (int(sim.x), int(sim.z)):
                print(f"  Frame {frame}: BLOCKED at ({sim.x:.0f}, {sim.z:.0f}) yaw={math.degrees(sim.yaw):.0f}deg")
                last_blocked_pos = (int(sim.x), int(sim.z))

            # Turn to try another direction
            if stuck_streak > 3:
                sim.move(3, debug=False)  # LEFT
                stuck_streak = 0
        else:
            stuck_streak = 0

        # Print doorway band every 20 frames
        if frame % 20 == 0 and frame > 0:
            print(f"\n=== Frame {frame} - Doorway band at current position ===")
            print(f"Position: ({sim.x:.0f}, {sim.z:.0f}) -> world ({world_x:.2f}, {world_y:.2f})")
            occ_map.print_obstacle_band(world_x, world_y, world_yaw)

    # Final statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)

    stats = occ_map.get_obstacle_stats()
    print(f"\nObstacle cells: {stats['total']}")
    print(f"  From depth sensing: {stats['from_depth']}")
    print(f"  From collision stamp: {stats['from_collision']}")

    map_stats = occ_map.get_stats()
    print(f"\nMap coverage:")
    print(f"  Unknown: {map_stats['unknown_pct']:.1f}%")
    print(f"  Free: {map_stats['free_pct']:.1f}%")
    print(f"  Occupied: {map_stats['occupied_pct']:.1f}%")

    # Check specific doorway region
    print("\n" + "=" * 70)
    print("DOORWAY REGION CHECK")
    print("=" * 70)

    # Check the doorway between Living Room and Hallway
    # In GLB coords: Living Room z > 225, Hallway z < 225
    # The doorway should be around z=225

    doorway_z = 225.0
    test_positions = [
        (-300.0, doorway_z + 20, 0, "From Living Room facing hallway"),
        (-300.0, doorway_z - 20, math.pi, "From Hallway facing living room"),
    ]

    for sim_x, sim_z, yaw, name in test_positions:
        world_x, world_y = sim_to_world(sim_x, sim_z)

        print(f"\n{name}:")
        print(f"  Sim pos: ({sim_x:.0f}, {sim_z:.0f}) -> World: ({world_x:.2f}, {world_y:.2f})")
        occ_map.print_obstacle_band(world_x, world_y, yaw)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_test()
