"""
Simple standalone test for backward escape behavior.
Tests if BACKWARD actually moves the drone.
"""

import sys
import math

# Add project root to path
sys.path.insert(0, '.')

from simulator.glb_simulator import GLBSimulator

def test_backward():
    print("=" * 60)
    print("BACKWARD ESCAPE TEST")
    print("=" * 60)

    # Initialize simulator
    print("\n1. Loading GLB simulator...")
    sim = GLBSimulator(
        glb_path="simulator/home_-_p3.glb",
        width=640,
        height=480
    )

    # Get initial position
    print(f"\n2. Initial position: x={sim.x:.2f}, z={sim.z:.2f}, yaw={math.degrees(sim.yaw):.1f}°")
    print(f"   Room: {sim._get_current_room()}")

    # Move forward a few times to get into a position
    print("\n3. Moving forward 5 times...")
    for i in range(5):
        old_x, old_z = sim.x, sim.z
        moved = sim.move(1, debug=False)  # Forward
        print(f"   Forward {i+1}: ({old_x:.2f},{old_z:.2f}) -> ({sim.x:.2f},{sim.z:.2f}) moved={moved}")

    # Now test backward
    print("\n4. Testing BACKWARD movement...")
    print("   Will try backward 5 times and report each move:")

    for i in range(5):
        old_x, old_z = sim.x, sim.z
        old_yaw = sim.yaw

        # Try backward (action 2)
        moved = sim.move(2, debug=True)

        # Calculate actual distance moved
        dx = sim.x - old_x
        dz = sim.z - old_z
        dist = math.sqrt(dx*dx + dz*dz)

        print(f"\n   Backward attempt {i+1}:")
        print(f"     Before: ({old_x:.2f}, {old_z:.2f})")
        print(f"     After:  ({sim.x:.2f}, {sim.z:.2f})")
        print(f"     Distance moved: {dist:.4f}")
        print(f"     Yaw: {math.degrees(old_yaw):.1f}° -> {math.degrees(sim.yaw):.1f}°")
        print(f"     Result: {'MOVED' if dist > 0.01 else 'DID NOT MOVE'}")

    # Try backing up after rotation
    print("\n5. Testing BACKWARD after 90° rotation...")
    sim.move(3)  # Turn left
    sim.move(3)  # Turn left
    sim.move(3)  # Turn left (30° each = 90° total? check turn_speed)

    for i in range(3):
        old_x, old_z = sim.x, sim.z
        moved = sim.move(2, debug=True)
        dx = sim.x - old_x
        dz = sim.z - old_z
        dist = math.sqrt(dx*dx + dz*dz)
        print(f"   After rotation, backward {i+1}: dist={dist:.4f} {'MOVED' if dist > 0.01 else 'STUCK'}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_backward()
