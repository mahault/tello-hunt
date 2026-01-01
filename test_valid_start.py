"""
Test with a valid starting position that's clearly in open space.

The issue is that (-200, 350) is on the Kitchen/Living Room wall boundary.
Let's find a valid starting position and then do the doorway diagnostic.
"""

import sys
import math
import numpy as np

sys.path.insert(0, '.')

from simulator.glb_simulator import GLBSimulator
from pomdp.frontier_explorer import FrontierExplorer
from utils.occupancy_map import OccupancyMap


def find_valid_start(sim):
    """Find a position where the drone can actually move."""
    # Try positions in the center of each room
    test_positions = [
        # Kitchen center (x=[-195, -9], z=[225, 441])
        (-100, 333, "Kitchen center"),
        (-150, 350, "Kitchen west"),
        (-50, 350, "Kitchen east"),

        # Living Room center (x=[-381, -195], z=[225, 441])
        (-290, 333, "Living Room center"),
        (-250, 350, "Living Room east"),
        (-330, 350, "Living Room west"),

        # Hallway center (x=[-381, -9], z=[9, 225])
        (-195, 117, "Hallway center"),
        (-150, 150, "Hallway east"),
        (-250, 150, "Hallway west"),
    ]

    for x, z, name in test_positions:
        sim.x, sim.z = x, z
        sim.yaw = 0.0

        # Try moving in 4 directions
        can_move = 0
        for yaw in [0, math.pi/2, math.pi, 3*math.pi/2]:
            sim.x, sim.z = x, z
            sim.yaw = yaw

            old_x, old_z = sim.x, sim.z
            moved = sim.move(1, debug=False)

            if abs(sim.x - old_x) > 5 or abs(sim.z - old_z) > 5:
                can_move += 1

            # Restore position for next test
            sim.x, sim.z = x, z

        print(f"  {name} ({x}, {z}): can move in {can_move}/4 directions")

        if can_move >= 2:
            return x, z, name

    return None, None, None


def print_doorway_band(occ_map, world_x, world_y, world_yaw, label=""):
    """Print grid values across doorway band."""
    grid = occ_map.grid
    world_to_map = occ_map.world_to_map

    fwd_dist = 0.6
    n_samples = 11
    lateral_range = 1.0

    print(f"\n{label}: ({world_x:.2f}, {world_y:.2f}) yaw={math.degrees(world_yaw):.0f}deg")

    values = []
    origins = []

    for i in range(n_samples):
        lateral = -0.5 + (i / (n_samples - 1)) * lateral_range

        fwd_x = math.cos(world_yaw)
        fwd_y = math.sin(world_yaw)
        right_x = math.cos(world_yaw - math.pi/2)
        right_y = math.sin(world_yaw - math.pi/2)

        sample_x = world_x + fwd_dist * fwd_x + lateral * right_x
        sample_y = world_y + fwd_dist * fwd_y + lateral * right_y

        map_x, map_y = world_to_map(sample_x, sample_y)

        if 0 <= map_x < grid.shape[1] and 0 <= map_y < grid.shape[0]:
            value = grid[map_y, map_x]
        else:
            value = -1

        values.append(value)

    # Visual representation
    visual = ""
    for v in values:
        if v < 0:
            visual += "?"
        elif v < 64:
            visual += "#"
        elif v < 150:
            visual += "."
        else:
            visual += " "

    n_obstacles = sum(1 for v in values if 0 <= v < 64)
    n_unknown = sum(1 for v in values if 64 <= v < 150)
    n_free = sum(1 for v in values if v >= 150)

    print(f"  Visual: [{visual}]  obs={n_obstacles} unk={n_unknown} free={n_free}")

    return values


def run_test():
    print("=" * 70)
    print("VALID START POSITION TEST")
    print("=" * 70)

    print("\n1. Loading simulator...")
    sim = GLBSimulator(glb_path="simulator/home_-_p3.glb", width=640, height=480)

    print("\n2. Finding valid starting position...")
    start_x, start_z, start_name = find_valid_start(sim)

    if start_x is None:
        print("ERROR: Could not find valid starting position!")
        return

    print(f"\n   Selected: {start_name} at ({start_x}, {start_z})")

    print("\n3. Initializing components...")
    occ_map = OccupancyMap()
    explorer = FrontierExplorer()

    # Set origin for coord transform
    origin_x, origin_z = start_x, start_z

    def sim_to_world(sim_x, sim_z):
        sx = (sim_x - origin_x) / 100.0
        sz = (sim_z - origin_z) / 100.0
        return -sz, sx

    print("\n4. Running exploration (200 frames)...")

    sim.x, sim.z = start_x, start_z
    sim.yaw = 0.0

    positions = [(sim.x, sim.z)]
    unique_positions = set()
    rooms_visited = set()

    for frame in range(200):
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

        # Get frontier action
        action = explorer.choose_action(
            grid=occ_map.grid,
            agent_x=world_x,
            agent_y=world_y,
            agent_yaw=world_yaw,
            world_to_map_fn=occ_map.world_to_map,
            map_to_world_fn=occ_map.map_to_world,
            debug=(frame % 50 == 0),
        )

        # Execute
        old_x, old_z = sim.x, sim.z
        if action > 0:
            sim.move(action, debug=False)

        # Track progress
        new_pos = (int(sim.x/20), int(sim.z/20))
        unique_positions.add(new_pos)
        positions.append((sim.x, sim.z))

        current_room = sim._get_current_room()
        if current_room != "Unknown":
            rooms_visited.add(current_room)

        # Progress report
        if frame % 50 == 0:
            dist = math.sqrt((sim.x - start_x)**2 + (sim.z - start_z)**2)
            print(f"  Frame {frame}: pos=({sim.x:.0f}, {sim.z:.0f}) room={current_room} "
                  f"unique={len(unique_positions)} dist={dist:.0f}")

    # Final position report
    print(f"\n5. Exploration complete!")
    print(f"   Rooms visited: {rooms_visited}")
    print(f"   Unique grid positions: {len(unique_positions)}")

    # Map stats
    stats = occ_map.get_stats()
    print(f"\n   Map coverage:")
    print(f"     Unknown: {stats['unknown_pct']:.1f}%")
    print(f"     Free: {stats['free_pct']:.1f}%")
    print(f"     Occupied: {stats['occupied_pct']:.1f}%")

    # Now check doorway bands at key positions
    print("\n" + "=" * 70)
    print("DOORWAY BAND DIAGNOSTIC")
    print("=" * 70)

    # Test positions near room transitions
    doorway_tests = [
        # Living Room <-> Hallway boundary at z=225
        {"sim_x": -290, "sim_z": 240, "yaw": math.pi, "name": "LR->Hallway (south)"},
        {"sim_x": -290, "sim_z": 210, "yaw": 0, "name": "Hallway->LR (north)"},

        # Kitchen <-> Hallway boundary
        {"sim_x": -100, "sim_z": 240, "yaw": math.pi, "name": "Kitchen->Hallway (south)"},
        {"sim_x": -100, "sim_z": 210, "yaw": 0, "name": "Hallway->Kitchen (north)"},

        # Living Room <-> Kitchen boundary at x=-195
        {"sim_x": -210, "sim_z": 333, "yaw": math.pi/2, "name": "LR->Kitchen (east)"},
        {"sim_x": -180, "sim_z": 333, "yaw": -math.pi/2, "name": "Kitchen->LR (west)"},
    ]

    for test in doorway_tests:
        sim.x = test["sim_x"]
        sim.z = test["sim_z"]
        world_x, world_y = sim_to_world(sim.x, sim.z)
        world_yaw = test["yaw"]

        print_doorway_band(occ_map, world_x, world_y, world_yaw, test["name"])

    # Check where obstacles are concentrated
    print("\n" + "=" * 70)
    print("OBSTACLE DISTRIBUTION")
    print("=" * 70)

    grid = occ_map.grid
    obstacle_mask = grid < 64
    n_obstacles = np.sum(obstacle_mask)

    print(f"\nTotal obstacle cells: {n_obstacles}")

    if n_obstacles > 0:
        # Find obstacle positions
        obstacle_ys, obstacle_xs = np.where(obstacle_mask)

        # Convert back to world coords
        print("\nObstacle locations (sample):")
        for i in range(min(10, len(obstacle_xs))):
            cx, cy = obstacle_xs[i], obstacle_ys[i]
            wx, wy = occ_map.map_to_world(cx, cy)
            # Convert world to sim coords
            sim_x = wy * 100.0 + origin_x
            sim_z = -wx * 100.0 + origin_z
            print(f"  Map({cx}, {cy}) -> World({wx:.2f}, {wy:.2f}) -> Sim({sim_x:.0f}, {sim_z:.0f})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_test()
