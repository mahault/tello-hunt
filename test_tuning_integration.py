"""
Integration test for frontier explorer tuning.

Tests all tuning features together with the actual simulator:
1. Escape frequency (delayed scan)
2. Doorway detection and bonus
3. Room transition tracking
4. Center-of-gap bias

Run: python test_tuning_integration.py
"""

import sys
import math
import numpy as np

sys.path.insert(0, '.')

from simulator.glb_simulator import GLBSimulator
from pomdp.frontier_explorer import FrontierExplorer, RoomTransitionTracker
from utils.occupancy_map import OccupancyMap


def run_test():
    print("=" * 70)
    print("TUNING INTEGRATION TEST")
    print("=" * 70)

    # 1. Load simulator
    print("\n1. Loading simulator...")
    sim = GLBSimulator(glb_path="simulator/home_-_p3.glb", width=640, height=480)

    # 2. Initialize components
    print("\n2. Initializing components...")
    occ_map = OccupancyMap()
    explorer = FrontierExplorer()

    print(f"   Doorway bonus: {explorer.doorway_bonus}m")
    print(f"   Doorway aspect min: {explorer.doorway_aspect_min}")

    # Start position (Kitchen center)
    start_x, start_z = -100, 333
    origin_x, origin_z = start_x, start_z

    def sim_to_world(sim_x, sim_z):
        sx = (sim_x - origin_x) / 100.0
        sz = (sim_z - origin_z) / 100.0
        return -sz, sx

    # 3. Run exploration
    print("\n3. Running exploration (100 frames)...")

    sim.x, sim.z = start_x, start_z
    sim.yaw = 0.0

    doorway_clusters = 0
    total_clusters = 0

    for frame in range(100):
        world_x, world_y = sim_to_world(sim.x, sim.z)
        world_yaw = sim.yaw

        # Get depth
        depth_map = sim.get_depth_buffer()
        valid_depths = depth_map[(depth_map > 0) & (depth_map < 10000)]
        max_depth = max(np.percentile(valid_depths, 95), 100.0) if len(valid_depths) > 0 else 300.0
        rel_depth = 1.0 - np.clip(depth_map / max_depth, 0, 1)
        rel_depth = rel_depth.astype(np.float32)

        # Update occupancy
        occ_map.update_from_depth(rel_depth, world_x, world_y, world_yaw, max_range=3.0)

        # Report room for transition tracking
        current_room = sim._get_current_room()
        transition = explorer.report_room(current_room, (world_x, world_y))

        # Get action
        action = explorer.choose_action(
            grid=occ_map.grid,
            agent_x=world_x,
            agent_y=world_y,
            agent_yaw=world_yaw,
            world_to_map_fn=occ_map.world_to_map,
            map_to_world_fn=occ_map.map_to_world,
            debug=(frame % 25 == 0),
        )

        # Track doorway clusters
        for cluster in explorer.clusters:
            total_clusters += 1
            if cluster.is_doorway:
                doorway_clusters += 1

        # Execute and track collision
        if action > 0:
            old_x, old_z = sim.x, sim.z
            sim.move(action, debug=False)

            # Check if movement was blocked
            moved = abs(sim.x - old_x) > 5 or abs(sim.z - old_z) > 5
            if action == 1 and not moved:
                # Forward was blocked - record it
                explorer.record_block(was_blocked=True)
                # Also mark collision in occupancy map
                occ_map.mark_obstacle_ahead(world_x, world_y, world_yaw)

        # Progress
        if frame % 25 == 0:
            print(f"  Frame {frame}: pos=({sim.x:.0f}, {sim.z:.0f}) room={current_room}")

    # 4. Report results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    room_stats = explorer.get_room_stats()
    print(f"\nRoom Transitions:")
    print(f"  Total transitions: {room_stats['total_transitions']}")
    print(f"  Rooms visited: {room_stats['rooms_visited']}")
    print(f"  Room list: {room_stats['room_list']}")
    print(f"  Unique doorways used: {room_stats['unique_doorways']}")

    print(f"\nDoorway Detection:")
    print(f"  Total clusters analyzed: {total_clusters}")
    print(f"  Doorway clusters: {doorway_clusters}")
    pct = (doorway_clusters / total_clusters * 100) if total_clusters > 0 else 0
    print(f"  Doorway percentage: {pct:.1f}%")

    # Map stats
    stats = occ_map.get_stats()
    print(f"\nMap Coverage:")
    print(f"  Free: {stats['free_pct']:.1f}%")
    print(f"  Unknown: {stats['unknown_pct']:.1f}%")
    print(f"  Occupied: {stats['occupied_pct']:.1f}%")

    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
