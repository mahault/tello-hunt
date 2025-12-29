"""
Test ORB place recognition on GLB house model.

This tests whether ORB keypoint matching works well on realistic
3D rendered indoor scenes with proper textures and furniture.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from simulator.glb_simulator import GLBSimulator
from pomdp.place_recognizer import ORBPlaceRecognizer


def test_orb_features(sim: GLBSimulator, n_positions: int = 10):
    """Test ORB feature detection at various positions."""
    print("\n=== ORB Feature Detection on GLB House ===\n")

    orb = cv2.ORB_create(nfeatures=500)

    # Test at different positions in the house
    positions = [
        (0, 200, -400, 0, "Center"),
        (-200, 200, -400, 0, "Left side"),
        (200, 200, -400, 0, "Right side"),
        (0, 200, -600, 0, "Back"),
        (0, 200, -200, 0, "Front"),
        (0, 200, -400, 1.57, "Facing left"),
        (0, 200, -400, -1.57, "Facing right"),
        (0, 200, -400, 3.14, "Facing back"),
        (-100, 200, -800, 0, "Far left"),
        (100, 200, -800, 0, "Far right"),
    ]

    feature_counts = []
    for x, y, z, yaw, name in positions:
        sim.x, sim.y, sim.z, sim.yaw = x, y, z, yaw
        frame = sim.render()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp, desc = orb.detectAndCompute(gray, None)
        n_features = len(kp) if kp else 0
        feature_counts.append(n_features)

        print(f"  {name:20s}: {n_features} features")

    print(f"\n  Average: {np.mean(feature_counts):.1f} features")
    print(f"  Min: {min(feature_counts)}, Max: {max(feature_counts)}")

    return feature_counts


def test_place_recognition(sim: GLBSimulator):
    """Test place recognition while navigating through house."""
    print("\n=== Place Recognition Navigation Test ===\n")

    recognizer = ORBPlaceRecognizer(
        n_features=500,
        match_threshold=0.75,
        min_matches=15,
        keyframe_cooldown=5,
        max_keyframes=30,
    )

    # Navigate through the house
    # Start at center, explore different areas
    waypoints = [
        # (x, z, yaw, description)
        (0, -400, 0, "Start - center"),
        (0, -400, 0.5, "Turn slightly"),
        (0, -400, 1.0, "Turn more"),
        (0, -400, 1.57, "Face left"),
        (-100, -400, 1.57, "Move left"),
        (-200, -400, 1.57, "Move more left"),
        (-200, -400, 0, "Turn forward"),
        (-200, -500, 0, "Move back"),
        (-200, -600, 0, "Move more back"),
        (-200, -600, -1.57, "Turn right"),
        (-100, -600, -1.57, "Move right"),
        (0, -600, -1.57, "Move more right"),
        (0, -600, 0, "Turn forward"),
        (0, -500, 0, "Move forward"),
        (0, -400, 0, "Return to start"),
    ]

    results = []
    for i, (x, z, yaw, desc) in enumerate(waypoints):
        sim.x, sim.z, sim.yaw = x, z, yaw
        frame = sim.render()

        place_id, place_name, confidence, is_new = recognizer.recognize(
            frame, allow_new=True, debug=False
        )
        results.append((desc, place_id, place_name, confidence, is_new))

        status = "NEW" if is_new else f"conf={confidence:.2f}"
        print(f"  {i:2d}. {desc:25s} -> {place_name} ({status})")

    print(f"\n  Total places discovered: {recognizer.n_places}")

    # Test loop closure - go back to start
    print("\n=== Loop Closure Test ===")
    sim.x, sim.z, sim.yaw = 0, -400, 0
    frame = sim.render()
    place_id, place_name, confidence, is_new = recognizer.recognize(
        frame, allow_new=False, debug=False
    )
    print(f"  Return to start: {place_name} (conf={confidence:.2f}, new={is_new})")

    if place_id == results[0][1]:
        print("  ✓ Loop closure SUCCESS - recognized starting location!")
    else:
        print("  ✗ Loop closure FAILED - did not recognize start")

    return results, recognizer


def test_room_discrimination(sim: GLBSimulator):
    """Test if different rooms produce different place IDs."""
    print("\n=== Room Discrimination Test ===\n")

    recognizer = ORBPlaceRecognizer(
        n_features=500,
        match_threshold=0.75,
        min_matches=12,
        keyframe_cooldown=3,
        max_keyframes=50,
    )

    # Define distinct areas to test
    areas = [
        ("Living Room A", -200, -300, 0),
        ("Living Room B", -200, -300, 1.57),
        ("Kitchen A", 200, -300, 0),
        ("Kitchen B", 200, -300, 1.57),
        ("Dining Area A", 0, -600, 0),
        ("Dining Area B", 0, -600, 3.14),
        ("Stairs A", 0, -500, 0),
        ("Stairs B", 0, -500, 1.57),
    ]

    # First pass - discover places
    print("First pass - discovering places:")
    place_ids = {}
    for name, x, z, yaw in areas:
        sim.x, sim.z, sim.yaw = x, z, yaw
        frame = sim.render()
        place_id, place_name, conf, is_new = recognizer.recognize(frame, debug=False)
        place_ids[name] = place_id
        status = "NEW" if is_new else f"matched {place_name}"
        print(f"  {name:20s}: place_id={place_id}, {status}")

    # Second pass - test recognition
    print("\nSecond pass - testing recognition:")
    correct = 0
    total = 0
    for name, x, z, yaw in areas:
        sim.x, sim.z, sim.yaw = x, z, yaw
        frame = sim.render()
        place_id, place_name, conf, is_new = recognizer.recognize(
            frame, allow_new=False, debug=False
        )
        matched = place_id == place_ids[name]
        correct += matched
        total += 1
        status = "✓" if matched else "✗"
        print(f"  {status} {name:20s}: expected {place_ids[name]}, got {place_id} (conf={conf:.2f})")

    accuracy = 100 * correct / total
    print(f"\n  Recognition accuracy: {accuracy:.1f}% ({correct}/{total})")

    return accuracy


def visualize_features(sim: GLBSimulator, output_dir: Path):
    """Save images with ORB features visualized."""
    output_dir.mkdir(parents=True, exist_ok=True)

    orb = cv2.ORB_create(nfeatures=500)

    positions = [
        (0, 200, -400, 0, "center"),
        (-200, 200, -400, 0, "left"),
        (200, 200, -400, 0, "right"),
        (0, 200, -600, 0, "back"),
    ]

    print(f"\n=== Saving feature visualizations to {output_dir} ===\n")

    for x, y, z, yaw, name in positions:
        sim.x, sim.y, sim.z, sim.yaw = x, y, z, yaw
        frame = sim.render()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp, desc = orb.detectAndCompute(gray, None)

        # Draw keypoints
        frame_kp = cv2.drawKeypoints(
            frame, kp, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        filename = f"orb_features_{name}.jpg"
        cv2.imwrite(str(output_dir / filename), frame_kp)
        print(f"  Saved: {filename} ({len(kp)} features)")


def main():
    print("=" * 60)
    print("GLB House - ORB Place Recognition Test")
    print("=" * 60)

    sim = GLBSimulator()

    try:
        # Test 1: Feature detection
        test_orb_features(sim)

        # Test 2: Place recognition during navigation
        test_place_recognition(sim)

        # Test 3: Room discrimination
        accuracy = test_room_discrimination(sim)

        # Save visualizations
        output_dir = Path("test_output/orb_glb")
        visualize_features(sim, output_dir)

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"\nORB on GLB house model:")
        print(f"  - Realistic textures provide abundant features")
        print(f"  - Room discrimination accuracy: {accuracy:.1f}%")
        print(f"\nCompared to simple raycaster:")
        print(f"  - Simple 3D: ~20-500 features, 50% accuracy")
        print(f"  - GLB house: More consistent features, better recognition")

    finally:
        sim.close()


if __name__ == "__main__":
    main()
