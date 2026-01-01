"""
Escape Strategy Test V3 - Simple and Correct

Uses actual simulator move() results to detect stuck conditions.
No custom raycast - just checks if position changed after move.
"""

import sys
import math
import numpy as np
from typing import Tuple, Optional

sys.path.insert(0, '.')

from simulator.glb_simulator import GLBSimulator

STAY, FORWARD, BACKWARD, LEFT, RIGHT = 0, 1, 2, 3, 4
NAMES = ['STAY', 'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']


def run_backward_strategy(sim: GLBSimulator, start_pos: Tuple[float, float, float],
                         max_frames: int = 80) -> dict:
    """Current strategy: alternate turns, then BACKWARD after 7 blocks."""
    sim.x, sim.z = start_pos[0], start_pos[1]
    sim.yaw = math.radians(start_pos[2])

    stuck_streak = 0
    cooldown_until = 0
    frame = 0

    positions = [(sim.x, sim.z)]
    actions = []

    for frame in range(max_frames):
        old_x, old_z = sim.x, sim.z

        # Decide action
        if stuck_streak <= 6:
            if frame < cooldown_until:
                action = LEFT if (frame % 2 == 0) else RIGHT
            else:
                action = FORWARD
        elif stuck_streak == 7:
            action = BACKWARD
            cooldown_until = frame + 15
        else:
            action = LEFT if (frame % 2 == 0) else RIGHT

        # Execute
        if action > 0:
            sim.move(action, debug=False)

        # Check if stuck
        moved = abs(sim.x - old_x) > 1 or abs(sim.z - old_z) > 1

        if action == FORWARD and not moved:
            stuck_streak += 1
        elif action == BACKWARD and moved:
            stuck_streak = 0  # Reset on successful backward
        elif moved:
            stuck_streak = max(0, stuck_streak - 1)

        positions.append((sim.x, sim.z))
        actions.append(NAMES[action])

    return analyze_results("BACKWARD", positions, actions, start_pos)


def run_local_frontier_strategy(sim: GLBSimulator, start_pos: Tuple[float, float, float],
                                max_frames: int = 80) -> dict:
    """Local frontier: when stuck, try all directions and go to one that works."""
    sim.x, sim.z = start_pos[0], start_pos[1]
    sim.yaw = math.radians(start_pos[2])

    stuck_streak = 0
    escape_mode = False
    escape_target_yaw = None
    escape_steps = 0

    positions = [(sim.x, sim.z)]
    actions = []

    for frame in range(max_frames):
        old_x, old_z = sim.x, sim.z
        old_yaw = sim.yaw

        # Enter escape mode if stuck
        if stuck_streak >= 4 and not escape_mode:
            # Scan 360 degrees to find a clear direction
            best_yaw = None
            best_dist = 0

            for angle_offset in range(0, 360, 22):
                test_yaw = math.radians(angle_offset)

                # Temporarily set yaw and try forward
                sim.yaw = test_yaw
                test_x, test_z = sim.x, sim.z
                sim.move(FORWARD, debug=False)

                dist = math.sqrt((sim.x - test_x)**2 + (sim.z - test_z)**2)

                if dist > best_dist:
                    best_dist = dist
                    best_yaw = test_yaw

                # Restore position
                sim.x, sim.z = old_x, old_z

            if best_dist > 1:
                escape_target_yaw = best_yaw
                escape_mode = True
                escape_steps = 0

            sim.yaw = old_yaw  # Restore yaw

        # Execute escape
        if escape_mode and escape_target_yaw is not None:
            heading_error = escape_target_yaw - sim.yaw
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi

            if abs(heading_error) > 0.2:
                action = LEFT if heading_error > 0 else RIGHT
            else:
                action = FORWARD
                escape_steps += 1
                if escape_steps >= 5:
                    escape_mode = False
                    stuck_streak = 0
        else:
            # Normal: try forward
            action = FORWARD

        # Execute
        if action > 0:
            sim.move(action, debug=False)

        # Check if stuck
        moved = abs(sim.x - old_x) > 1 or abs(sim.z - old_z) > 1

        if action == FORWARD and not moved:
            stuck_streak += 1
        elif moved:
            stuck_streak = 0

        positions.append((sim.x, sim.z))
        actions.append(NAMES[action])

    return analyze_results("LOCAL_FRONTIER", positions, actions, start_pos)


def run_rotate_and_go_strategy(sim: GLBSimulator, start_pos: Tuple[float, float, float],
                               max_frames: int = 80) -> dict:
    """Rotate-and-go: when stuck, rotate until a direction is clear, then go."""
    sim.x, sim.z = start_pos[0], start_pos[1]
    sim.yaw = math.radians(start_pos[2])

    stuck_streak = 0
    escape_mode = False
    rotations_tried = 0
    escape_steps = 0

    positions = [(sim.x, sim.z)]
    actions = []

    for frame in range(max_frames):
        old_x, old_z = sim.x, sim.z

        # Enter escape mode if stuck
        if stuck_streak >= 4 and not escape_mode:
            escape_mode = True
            rotations_tried = 0
            escape_steps = 0

        # Execute escape
        if escape_mode:
            if rotations_tried < 16:  # Full rotation
                # Try forward to test this direction
                sim.move(FORWARD, debug=False)
                moved = abs(sim.x - old_x) > 1 or abs(sim.z - old_z) > 1

                if moved:
                    # Found a clear direction!
                    escape_steps += 1
                    if escape_steps >= 4:
                        escape_mode = False
                        stuck_streak = 0
                    action = FORWARD
                else:
                    # Restore and rotate
                    sim.x, sim.z = old_x, old_z
                    action = LEFT
                    rotations_tried += 1
            else:
                # Tried full rotation, nothing works - just keep rotating
                action = LEFT
                escape_mode = False
        else:
            action = FORWARD

        # Execute (if not already moved during test)
        if action == LEFT or action == RIGHT:
            sim.move(action, debug=False)
        elif action == FORWARD and escape_mode and escape_steps == 0:
            pass  # Already tried above
        elif action == FORWARD:
            sim.move(action, debug=False)

        # Check final position
        final_moved = abs(sim.x - old_x) > 1 or abs(sim.z - old_z) > 1

        if action == FORWARD and not final_moved:
            stuck_streak += 1
        elif final_moved:
            stuck_streak = max(0, stuck_streak - 1)

        positions.append((sim.x, sim.z))
        actions.append(NAMES[action])

    return analyze_results("ROTATE_AND_GO", positions, actions, start_pos)


def analyze_results(name: str, positions: list, actions: list, start: tuple) -> dict:
    """Analyze test results."""
    start_pos = positions[0]
    end_pos = positions[-1]

    dist = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
    unique = len(set((int(p[0]/10), int(p[1]/10)) for p in positions))

    # Count stuck (FORWARD with no position change)
    stuck = 0
    for i in range(1, len(positions)):
        if actions[i-1] == 'FORWARD':
            if abs(positions[i][0] - positions[i-1][0]) < 1 and abs(positions[i][1] - positions[i-1][1]) < 1:
                stuck += 1

    escaped = dist > 50  # More than 50 sim units = escaped

    counts = {a: actions.count(a) for a in set(actions)}

    return {
        'name': name,
        'distance': dist,
        'unique_positions': unique,
        'stuck_frames': stuck,
        'escaped': escaped,
        'action_counts': counts,
        'final_pos': end_pos,
    }


def main():
    print("=" * 70)
    print("ESCAPE STRATEGY TEST V3 (actual move results)")
    print("=" * 70)

    print("\n1. Loading simulator...")
    sim = GLBSimulator(glb_path="simulator/home_-_p3.glb", width=640, height=480)

    # Test positions: corners in different rooms
    corners = [
        (-298.0, 280.0, 210, "Living Room corner (facing wall)"),
        (-350.0, 350.0, 180, "Living Room center-ish"),
        (-280.0, 400.0, 90, "Living Room open area"),
    ]

    all_results = []

    for corner in corners:
        pos = corner[:3]
        name = corner[3]

        print(f"\n{'='*70}")
        print(f"SCENARIO: {name}")
        print(f"Start: ({pos[0]:.0f}, {pos[1]:.0f}) yaw={pos[2]}deg")
        print("-" * 70)

        # Run each strategy
        r1 = run_backward_strategy(sim, pos)
        r2 = run_local_frontier_strategy(sim, pos)
        r3 = run_rotate_and_go_strategy(sim, pos)

        for r in [r1, r2, r3]:
            r['scenario'] = name
            all_results.append(r)

            esc = "YES" if r['escaped'] else "no"
            print(f"  {r['name']:<18}: dist={r['distance']:>6.1f}  "
                  f"unique={r['unique_positions']:>3}  stuck={r['stuck_frames']:>3}  "
                  f"escaped={esc:>3}")

    # Summary
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)

    strategies = ['BACKWARD', 'LOCAL_FRONTIER', 'ROTATE_AND_GO']

    print(f"\n{'Strategy':<18} {'TotalDist':>10} {'Escapes':>8} {'TotalStuck':>12}")
    print("-" * 50)

    for s in strategies:
        results = [r for r in all_results if r['name'] == s]
        total_dist = sum(r['distance'] for r in results)
        escapes = sum(1 for r in results if r['escaped'])
        total_stuck = sum(r['stuck_frames'] for r in results)

        print(f"{s:<18} {total_dist:>10.1f} {escapes:>8}/3 {total_stuck:>12}")

    # Winner
    scores = {}
    for s in strategies:
        results = [r for r in all_results if r['name'] == s]
        score = (sum(1 for r in results if r['escaped']) * 100 +
                 sum(r['distance'] for r in results) -
                 sum(r['stuck_frames'] for r in results) * 0.5)
        scores[s] = score

    best = max(scores.items(), key=lambda x: x[1])
    print(f"\nBEST: {best[0]} (score: {best[1]:.1f})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
