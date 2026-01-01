"""
Standalone test for chokepoint escape behavior.
Tests if the drone can escape from a corner situation.
"""

import sys
import math
import numpy as np

# Add project root to path
sys.path.insert(0, '.')

from simulator.glb_simulator import GLBSimulator
from pomdp.frontier_explorer import FrontierExplorer
from utils.occupancy_map import OccupancyMap

def test_chokepoint_escape():
    print("=" * 60)
    print("CHOKEPOINT ESCAPE TEST")
    print("=" * 60)

    # Initialize simulator
    print("\n1. Loading GLB simulator...")
    sim = GLBSimulator(
        glb_path="simulator/home_-_p3.glb",
        width=640,
        height=480
    )

    # Initialize occupancy map and frontier explorer
    print("2. Initializing occupancy map and frontier explorer...")
    occ_map = OccupancyMap()  # Use defaults
    explorer = FrontierExplorer()

    print(f"\n3. Initial position: x={sim.x:.2f}, z={sim.z:.2f}, yaw={math.degrees(sim.yaw):.1f}°")
    print(f"   Room: {sim._get_current_room()}")

    # Move drone into a corner (Living Room corner)
    print("\n4. Moving drone into corner position...")
    # Set position near the corner of Living Room
    sim.x = -298.0  # Near left wall
    sim.z = 280.0   # Near back
    sim.yaw = math.radians(210)  # Facing into corner

    print(f"   Corner position: x={sim.x:.2f}, z={sim.z:.2f}, yaw={math.degrees(sim.yaw):.1f}°")
    print(f"   Room: {sim._get_current_room()}")

    # Create artificial obstacle in grid (simulating detected wall ahead)
    print("\n5. Marking obstacles in occupancy grid...")
    # Convert sim position to world coordinates
    start_x = sim.model_center[0] if hasattr(sim, 'model_center') else -102
    start_z = sim.model_center[2] if hasattr(sim, 'model_center') else 333
    world_x = -(sim.z - start_z) / 165.0
    world_y = -(sim.x - start_x) / 165.0
    world_yaw = sim.yaw

    print(f"   World position: ({world_x:.2f}, {world_y:.2f})")

    # Mark obstacles ahead of the drone (directly set grid cells to occupied = 0)
    for dist in [0.3, 0.4, 0.5, 0.6]:
        obs_x = world_x + math.cos(world_yaw) * dist
        obs_y = world_y + math.sin(world_yaw) * dist
        cx, cy = occ_map.world_to_map(obs_x, obs_y)
        if 0 <= cx < occ_map.grid.shape[1] and 0 <= cy < occ_map.grid.shape[0]:
            occ_map.grid[cy, cx] = 0  # Mark as occupied
            print(f"   Marked obstacle at ({obs_x:.2f}, {obs_y:.2f}) -> cell ({cx}, {cy})")

    # Run escape sequence
    print("\n6. Running escape sequence...")
    print("   Will run 50 frames and track position/actions")

    positions = []
    actions = []

    for frame in range(50):
        # Get current world pose
        world_x = -(sim.z - start_z) / 165.0
        world_y = -(sim.x - start_x) / 165.0
        world_yaw = sim.yaw

        # Choose action
        action = explorer.choose_action(
            grid=occ_map.grid,
            agent_x=world_x,
            agent_y=world_y,
            agent_yaw=world_yaw,
            world_to_map_fn=occ_map.world_to_map,
            map_to_world_fn=occ_map.map_to_world,
            debug=(frame % 10 == 0),
        )

        action_names = ['STAY', 'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']
        action_name = action_names[action] if action < len(action_names) else f'ACTION_{action}'

        # Execute action
        old_x, old_z = sim.x, sim.z
        if action > 0:
            moved = sim.move(action, debug=False)
        else:
            moved = True

        # Track
        positions.append((sim.x, sim.z))
        actions.append(action_name)

        # Print key frames
        if action == 2:  # BACKWARD
            print(f"   Frame {frame}: {action_name} ({old_x:.1f},{old_z:.1f}) -> ({sim.x:.1f},{sim.z:.1f})")
        elif frame % 10 == 0:
            print(f"   Frame {frame}: {action_name} at ({sim.x:.1f},{sim.z:.1f})")

    # Analyze results
    print("\n7. Analysis:")

    # Count actions
    action_counts = {}
    for a in actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    print("   Action distribution:")
    for a, c in sorted(action_counts.items()):
        print(f"     {a}: {c}")

    # Check if position changed significantly
    start_pos = positions[0]
    end_pos = positions[-1]
    total_dist = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
    print(f"\n   Total distance moved: {total_dist:.2f}")
    print(f"   Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f})")
    print(f"   End: ({end_pos[0]:.1f}, {end_pos[1]:.1f})")

    # Check if BACKWARD was followed by rotation (not immediate FORWARD)
    backward_followed_by = []
    for i, a in enumerate(actions):
        if a == 'BACKWARD' and i + 1 < len(actions):
            next_action = actions[i + 1]
            backward_followed_by.append(next_action)

    print(f"\n   Actions after BACKWARD: {backward_followed_by}")
    forward_after_backward = sum(1 for a in backward_followed_by if a == 'FORWARD')
    print(f"   FORWARD immediately after BACKWARD: {forward_after_backward}")

    if forward_after_backward == 0:
        print("\n   ✓ PASS: No immediate FORWARD after BACKWARD (cooldown working)")
    else:
        print("\n   ✗ FAIL: FORWARD occurred immediately after BACKWARD")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_chokepoint_escape()
