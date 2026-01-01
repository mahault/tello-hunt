"""
Test the new hybrid escape strategy.

The hybrid escape:
1. When stuck (streak >= 5), FrontierExplorer sets needs_escape_scan = True
2. Pipeline scans 360° using actual sim.move() to find clear direction
3. Pipeline calls set_escape_direction(best_yaw)
4. FrontierExplorer rotates toward that yaw and moves forward
"""

import sys
import math
import numpy as np

sys.path.insert(0, '.')

from simulator.glb_simulator import GLBSimulator
from pomdp.frontier_explorer import FrontierExplorer
from utils.occupancy_map import OccupancyMap


def scan_for_escape(sim, original_x, original_z, original_yaw):
    """Scan 360 degrees to find best escape direction."""
    best_yaw = original_yaw
    best_dist = 0.0

    for angle_idx in range(16):
        test_yaw = angle_idx * (2 * math.pi / 16)
        sim.yaw = test_yaw
        test_x, test_z = sim.x, sim.z

        sim.move(1, debug=False)  # FORWARD

        dist = math.sqrt((sim.x - test_x)**2 + (sim.z - test_z)**2)
        if dist > best_dist:
            best_dist = dist
            best_yaw = test_yaw

        sim.x, sim.z = original_x, original_z

    sim.x, sim.z, sim.yaw = original_x, original_z, original_yaw
    return best_yaw, best_dist


def run_hybrid_escape_test(sim, occ_map, explorer, start_pos, max_frames=80):
    """Test hybrid escape from a corner position."""

    sim.x, sim.z = start_pos[0], start_pos[1]
    sim.yaw = math.radians(start_pos[2])
    explorer.reset()

    # Coordinate conversion
    start_x = sim.model_center[0] if hasattr(sim, 'model_center') else -102
    start_z = sim.model_center[2] if hasattr(sim, 'model_center') else 333

    def sim_to_world(sim_x, sim_z):
        return -(sim_z - start_z) / 165.0, -(sim_x - start_x) / 165.0

    positions = [(sim.x, sim.z)]
    actions = []

    for frame in range(max_frames):
        world_x, world_y = sim_to_world(sim.x, sim.z)
        world_yaw = sim.yaw

        # Get action from explorer
        action = explorer.choose_action(
            grid=occ_map.grid,
            agent_x=world_x,
            agent_y=world_y,
            agent_yaw=world_yaw,
            world_to_map_fn=occ_map.world_to_map,
            map_to_world_fn=occ_map.map_to_world,
            debug=(frame % 20 == 0),
        )

        # Check if escape scan needed
        if explorer.needs_escape_scan:
            print(f"  Frame {frame}: Escape scan requested")
            best_yaw, best_dist = scan_for_escape(sim, sim.x, sim.z, sim.yaw)
            print(f"  Frame {frame}: Best escape: yaw={math.degrees(best_yaw):.0f}° dist={best_dist:.1f}")

            if best_dist > 5.0:
                explorer.set_escape_direction(best_yaw)
            else:
                print(f"  Frame {frame}: No clear direction found")
                explorer.cancel_escape()

        # Execute action
        action_names = ['STAY', 'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']
        action_name = action_names[action] if action < len(action_names) else f'ACT{action}'

        old_x, old_z = sim.x, sim.z
        if action > 0:
            sim.move(action, debug=False)

        # Log key events
        if action_name in ['FORWARD', 'BACKWARD'] or frame % 20 == 0:
            moved = math.sqrt((sim.x - old_x)**2 + (sim.z - old_z)**2)
            if moved > 1 or frame % 20 == 0:
                print(f"  Frame {frame}: {action_name} -> ({sim.x:.0f}, {sim.z:.0f}) moved={moved:.1f}")

        positions.append((sim.x, sim.z))
        actions.append(action_name)

    # Analyze results
    start_p = positions[0]
    end_p = positions[-1]
    total_dist = math.sqrt((end_p[0] - start_p[0])**2 + (end_p[1] - start_p[1])**2)
    unique_pos = len(set((int(p[0]/10), int(p[1]/10)) for p in positions))

    stuck = sum(1 for i in range(1, len(positions))
                if actions[i-1] == 'FORWARD' and
                abs(positions[i][0] - positions[i-1][0]) < 1 and
                abs(positions[i][1] - positions[i-1][1]) < 1)

    escaped = total_dist > 50

    return {
        'distance': total_dist,
        'unique_positions': unique_pos,
        'stuck_frames': stuck,
        'escaped': escaped,
        'action_counts': {a: actions.count(a) for a in set(actions)},
    }


def main():
    print("=" * 70)
    print("HYBRID ESCAPE STRATEGY TEST")
    print("=" * 70)

    print("\n1. Loading simulator...")
    sim = GLBSimulator(glb_path="simulator/home_-_p3.glb", width=640, height=480)

    print("2. Initializing components...")
    occ_map = OccupancyMap()
    explorer = FrontierExplorer()

    # Test corners
    corners = [
        (-298.0, 280.0, 210, "Living Room corner"),
        (-350.0, 350.0, 180, "Living Room toward wall"),
        (-280.0, 400.0, 90, "Living Room open"),
    ]

    results = []

    for corner in corners:
        pos = corner[:3]
        name = corner[3]

        print(f"\n{'='*70}")
        print(f"SCENARIO: {name}")
        print(f"Start: ({pos[0]:.0f}, {pos[1]:.0f}) yaw={pos[2]}°")
        print("-" * 70)

        result = run_hybrid_escape_test(sim, occ_map, explorer, pos, max_frames=80)
        result['scenario'] = name
        results.append(result)

        esc = "YES" if result['escaped'] else "no"
        print(f"\nResult: dist={result['distance']:.1f}  unique={result['unique_positions']}  "
              f"stuck={result['stuck_frames']}  escaped={esc}")
        print(f"Actions: {result['action_counts']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_dist = sum(r['distance'] for r in results)
    total_escapes = sum(1 for r in results if r['escaped'])
    total_stuck = sum(r['stuck_frames'] for r in results)

    print(f"\nTotal distance: {total_dist:.1f}")
    print(f"Escapes: {total_escapes}/3")
    print(f"Total stuck frames: {total_stuck}")

    if total_escapes >= 2:
        print("\nHYBRID ESCAPE: WORKING WELL")
    elif total_escapes >= 1:
        print("\nHYBRID ESCAPE: PARTIALLY WORKING")
    else:
        print("\nHYBRID ESCAPE: NEEDS IMPROVEMENT")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
