"""
Doorway Diagnostic Test

When the drone is stuck at a room boundary, this test prints the grid values
across the likely doorway band (11 cells across a 1m line perpendicular to
heading at 0.6m forward).

If we see a "wall of zeros" where the doorway should be, that explains why
the drone can't find a path through.
"""

import sys
import math
import numpy as np

sys.path.insert(0, '.')

from simulator.glb_simulator import GLBSimulator
from pomdp.frontier_explorer import FrontierExplorer
from utils.occupancy_map import OccupancyMap


def print_doorway_band(occ_map, world_x, world_y, world_yaw, label=""):
    """
    Print grid values across a 1m line perpendicular to heading at 0.6m forward.

    Samples 11 cells spanning from -0.5m to +0.5m lateral, at 0.6m forward.
    """
    grid = occ_map.grid
    world_to_map = occ_map.world_to_map

    print(f"\n{'='*60}")
    print(f"DOORWAY BAND DIAGNOSTIC {label}")
    print(f"{'='*60}")
    print(f"Agent position: ({world_x:.2f}, {world_y:.2f}) yaw={math.degrees(world_yaw):.0f}deg")

    # Forward offset (0.6m ahead)
    fwd_dist = 0.6

    # Sample 11 points from -0.5m to +0.5m lateral
    n_samples = 11
    lateral_range = 1.0  # Total span in meters

    print(f"\nSampling {n_samples} cells across {lateral_range}m at {fwd_dist}m forward:")
    print(f"{'Lateral':>8} {'WorldX':>8} {'WorldY':>8} {'MapX':>6} {'MapY':>6} {'Value':>6} {'Status':>10}")
    print("-" * 60)

    values = []
    for i in range(n_samples):
        # Lateral offset from -0.5 to +0.5
        lateral = -0.5 + (i / (n_samples - 1)) * lateral_range

        # Calculate world position
        # Forward direction
        fwd_x = math.cos(world_yaw)
        fwd_y = math.sin(world_yaw)
        # Right direction (perpendicular)
        right_x = math.cos(world_yaw - math.pi/2)
        right_y = math.sin(world_yaw - math.pi/2)

        sample_x = world_x + fwd_dist * fwd_x + lateral * right_x
        sample_y = world_y + fwd_dist * fwd_y + lateral * right_y

        # Convert to map coordinates
        map_x, map_y = world_to_map(sample_x, sample_y)

        # Get grid value
        if 0 <= map_x < grid.shape[1] and 0 <= map_y < grid.shape[0]:
            value = grid[map_y, map_x]
        else:
            value = -1  # Out of bounds

        values.append(value)

        # Determine status
        if value == -1:
            status = "OUT"
        elif value < 64:
            status = "OBSTACLE"
        elif value < 150:
            status = "unknown"
        else:
            status = "free"

        print(f"{lateral:>8.2f} {sample_x:>8.2f} {sample_y:>8.2f} {map_x:>6} {map_y:>6} {value:>6} {status:>10}")

    # Summary
    n_obstacles = sum(1 for v in values if 0 <= v < 64)
    n_unknown = sum(1 for v in values if 64 <= v < 150)
    n_free = sum(1 for v in values if v >= 150)
    n_out = sum(1 for v in values if v < 0)

    print("-" * 60)
    print(f"Summary: obstacles={n_obstacles}, unknown={n_unknown}, free={n_free}, out={n_out}")

    # Visual representation
    visual = ""
    for v in values:
        if v < 0:
            visual += "?"
        elif v < 64:
            visual += "#"  # Obstacle
        elif v < 150:
            visual += "."  # Unknown
        else:
            visual += " "  # Free
    print(f"Visual: [{visual}]  (# = obstacle, . = unknown, space = free)")

    # Diagnosis
    if n_obstacles >= 5:
        print("\nDIAGNOSIS: Wall of obstacles in doorway band!")
        print("This explains why the drone can't path through.")
    elif n_unknown >= 8:
        print("\nDIAGNOSIS: Mostly unknown - needs more depth observations")
    elif n_free >= 5:
        print("\nDIAGNOSIS: Doorway appears open - path should be possible")

    return values


def run_diagnostic():
    print("=" * 70)
    print("DOORWAY DIAGNOSTIC TEST")
    print("=" * 70)

    print("\n1. Loading simulator...")
    sim = GLBSimulator(glb_path="simulator/home_-_p3.glb", width=640, height=480)

    print("2. Initializing occupancy map...")
    occ_map = OccupancyMap()

    print("3. Initializing frontier explorer...")
    explorer = FrontierExplorer()

    # Set local origin
    origin_x, origin_z = sim.x, sim.z

    def sim_to_world(sim_x, sim_z):
        """Convert GLB sim coords to world coords."""
        sx = (sim_x - origin_x) / 100.0
        sz = (sim_z - origin_z) / 100.0
        world_x = -sz  # forward = -Z in GLB
        world_y = sx   # right = +X in GLB
        return world_x, world_y

    # Test positions at room boundaries
    # These are the corners where the drone tends to get stuck
    test_positions = [
        # Living Room edge facing hallway/kitchen
        {"sim_x": -298.0, "sim_z": 280.0, "yaw_deg": 210, "name": "Living Room corner A"},
        {"sim_x": -350.0, "sim_z": 350.0, "yaw_deg": 180, "name": "Living Room center"},
        {"sim_x": -280.0, "sim_z": 400.0, "yaw_deg": 90, "name": "Living Room toward hall"},
        # After some exploration - typical stuck spots
        {"sim_x": -300.0, "sim_z": 320.0, "yaw_deg": 0, "name": "Living Room north edge"},
        {"sim_x": -320.0, "sim_z": 300.0, "yaw_deg": 270, "name": "Living Room west edge"},
    ]

    # Run exploration for a bit to build up the occupancy map
    print("\n4. Running exploration to build map (50 frames)...")

    sim.x, sim.z = -300.0, 350.0
    sim.yaw = 0.0

    for frame in range(50):
        world_x, world_y = sim_to_world(sim.x, sim.z)
        world_yaw = sim.yaw

        # Get depth and update map
        depth_map = sim.get_depth_buffer()
        valid_depths = depth_map[(depth_map > 0) & (depth_map < 10000)]
        if len(valid_depths) > 0:
            max_depth = max(np.percentile(valid_depths, 95), 100.0)
        else:
            max_depth = 300.0
        rel_depth = 1.0 - np.clip(depth_map / max_depth, 0, 1)
        rel_depth = rel_depth.astype(np.float32)

        # Update occupancy
        occ_map.update_from_depth(rel_depth, world_x, world_y, world_yaw, max_range=3.0)

        # Get action from explorer
        action = explorer.choose_action(
            grid=occ_map.grid,
            agent_x=world_x,
            agent_y=world_y,
            agent_yaw=world_yaw,
            world_to_map_fn=occ_map.world_to_map,
            map_to_world_fn=occ_map.map_to_world,
            debug=False,
        )

        # Execute
        if action > 0:
            sim.move(action, debug=False)

        # Print progress
        if frame % 10 == 0:
            print(f"  Frame {frame}: pos=({sim.x:.0f}, {sim.z:.0f}) action={action}")

    # Now check doorway bands at each test position
    print("\n5. Checking doorway bands at test positions...")

    for pos in test_positions:
        sim.x = pos["sim_x"]
        sim.z = pos["sim_z"]
        sim.yaw = math.radians(pos["yaw_deg"])

        world_x, world_y = sim_to_world(sim.x, sim.z)
        world_yaw = sim.yaw

        print_doorway_band(occ_map, world_x, world_y, world_yaw, label=pos["name"])

    # Also check current exploration state
    print("\n" + "=" * 70)
    print("EXPLORATION STATE CHECK")
    print("=" * 70)

    # Move to a typical stuck position and show what explorer sees
    sim.x, sim.z = -298.0, 280.0
    sim.yaw = math.radians(210)

    world_x, world_y = sim_to_world(sim.x, sim.z)
    world_yaw = sim.yaw

    print(f"\nAgent at Living Room corner: ({world_x:.2f}, {world_y:.2f}) yaw={math.degrees(world_yaw):.0f}deg")

    # Get action with debug
    print("\nExplorer decision:")
    explorer.reset()
    action = explorer.choose_action(
        grid=occ_map.grid,
        agent_x=world_x,
        agent_y=world_y,
        agent_yaw=world_yaw,
        world_to_map_fn=occ_map.world_to_map,
        map_to_world_fn=occ_map.map_to_world,
        debug=True,
    )

    action_names = ['STAY', 'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']
    print(f"\nChosen action: {action_names[action] if action < len(action_names) else action}")

    # Check if there's a frontier target
    if explorer.target:
        tx, ty = explorer.target
        print(f"Current target: ({tx:.2f}, {ty:.2f})")

        # Distance to target
        dist = math.sqrt((tx - world_x)**2 + (ty - world_y)**2)
        print(f"Distance to target: {dist:.2f}m")

        # Direction to target
        target_dir = math.atan2(ty - world_y, tx - world_x)
        heading_error = target_dir - world_yaw
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        print(f"Heading error: {math.degrees(heading_error):.0f}deg")
    else:
        print("No current target")

    # Show map statistics
    stats = occ_map.get_stats()
    print(f"\nMap stats:")
    print(f"  Unknown: {stats['unknown_pct']:.1f}%")
    print(f"  Free: {stats['free_pct']:.1f}%")
    print(f"  Occupied: {stats['occupied_pct']:.1f}%")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostic()
