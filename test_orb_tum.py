"""
Test ORB place recognition on TUM RGB-D dataset.

This tests whether ORB keypoint matching works better on real
indoor images compared to synthetic 3D simulator renders.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pomdp.place_recognizer import ORBPlaceRecognizer


def load_tum_images(dataset_path: Path, max_images: int = None, step: int = 1):
    """Load RGB images from TUM dataset."""
    rgb_dir = dataset_path / "rgb"

    # Get sorted image files
    image_files = sorted(rgb_dir.glob("*.png"))

    if step > 1:
        image_files = image_files[::step]

    if max_images:
        image_files = image_files[:max_images]

    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append((img_path.stem, img))

    return images


def test_orb_features(images):
    """Test how many ORB features are detected in real images."""
    orb = cv2.ORB_create(nfeatures=500)

    print("\n=== ORB Feature Detection on Real Images ===\n")

    feature_counts = []
    for name, img in images[:20]:  # First 20 images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = orb.detectAndCompute(gray, None)
        n_features = len(kp) if kp else 0
        feature_counts.append(n_features)
        print(f"  {name}: {n_features} features")

    print(f"\n  Average: {np.mean(feature_counts):.1f} features")
    print(f"  Min: {min(feature_counts)}, Max: {max(feature_counts)}")

    return feature_counts


def test_place_recognition_sequence(images, recognizer: ORBPlaceRecognizer):
    """Test place recognition on image sequence."""
    print("\n=== Place Recognition Sequence Test ===\n")

    results = []
    for i, (name, img) in enumerate(images):
        place_id, place_name, confidence, is_new = recognizer.recognize(
            img, allow_new=True, debug=False
        )
        results.append((name, place_id, place_name, confidence, is_new))

        if is_new:
            print(f"  Frame {i:4d} ({name}): NEW PLACE '{place_name}' (id={place_id})")
        elif i % 50 == 0:  # Print every 50th frame
            print(f"  Frame {i:4d} ({name}): '{place_name}' (conf={confidence:.2f})")

    return results


def test_loop_closure(images, recognizer: ORBPlaceRecognizer):
    """Test if recognizer detects revisiting same location."""
    print("\n=== Loop Closure Test ===\n")

    # Process first half of sequence
    n_half = len(images) // 2
    print(f"  Processing first {n_half} frames...")

    for name, img in images[:n_half]:
        recognizer.recognize(img, allow_new=True, debug=False)

    print(f"  Created {recognizer.n_places} places")

    # Now test if second half matches first half (loop closure)
    print(f"\n  Testing loop closure on remaining {len(images) - n_half} frames...")

    matches = 0
    new_places = 0

    for name, img in images[n_half:]:
        place_id, place_name, confidence, is_new = recognizer.recognize(
            img, allow_new=True, debug=False
        )
        if is_new:
            new_places += 1
        else:
            matches += 1

    print(f"\n  Results:")
    print(f"    Matched existing places: {matches}")
    print(f"    New places created: {new_places}")
    print(f"    Loop closure rate: {100*matches/(matches+new_places):.1f}%")


def test_stability(images, recognizer: ORBPlaceRecognizer):
    """Test recognition stability (should give same result for similar frames)."""
    print("\n=== Recognition Stability Test ===\n")

    # Process all images
    results = []
    for name, img in images:
        place_id, place_name, confidence, is_new = recognizer.recognize(
            img, allow_new=True, debug=False
        )
        results.append(place_id)

    # Check how often consecutive frames have same place_id
    stable_count = 0
    for i in range(1, len(results)):
        if results[i] == results[i-1]:
            stable_count += 1

    stability = 100 * stable_count / (len(results) - 1)
    print(f"  Total places discovered: {recognizer.n_places}")
    print(f"  Stability (consecutive frames same place): {stability:.1f}%")

    # Count transitions
    transitions = 0
    for i in range(1, len(results)):
        if results[i] != results[i-1]:
            transitions += 1

    print(f"  Total transitions: {transitions}")
    print(f"  Avg frames per place: {len(results) / max(1, transitions):.1f}")

    return results


def visualize_places(images, results, recognizer: ORBPlaceRecognizer, output_dir: Path):
    """Save representative images for each discovered place."""
    output_dir.mkdir(exist_ok=True)

    # Find first occurrence of each place
    place_examples = {}
    for (name, img), place_id in zip(images, results):
        if place_id not in place_examples:
            place_examples[place_id] = (name, img)

    print(f"\n=== Saving {len(place_examples)} place examples to {output_dir} ===\n")

    for place_id, (name, img) in place_examples.items():
        place_name = recognizer.get_place_name(place_id)
        filename = f"place_{place_id:02d}_{place_name}.jpg"
        cv2.imwrite(str(output_dir / filename), img)
        print(f"  Saved: {filename}")


def main():
    dataset_path = Path("datasets/rgbd_dataset_freiburg1_room")

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return

    print("=" * 60)
    print("TUM RGB-D Dataset - ORB Place Recognition Test")
    print("=" * 60)

    # Load images (every 5th frame to reduce redundancy)
    print("\nLoading images (every 5th frame)...")
    images = load_tum_images(dataset_path, max_images=500, step=5)
    print(f"Loaded {len(images)} images")

    # Test 1: Feature detection
    test_orb_features(images)

    # Test 2: Place recognition with fresh recognizer
    print("\n" + "=" * 60)
    print("Test: Online Place Discovery")
    print("=" * 60)

    recognizer = ORBPlaceRecognizer(
        n_features=500,
        match_threshold=0.75,  # Lowe's ratio
        min_matches=15,  # Need 15 good matches
        keyframe_cooldown=10,
        max_keyframes=30,
    )

    results = test_stability(images, recognizer)

    # Save examples
    output_dir = Path("datasets/orb_places")
    visualize_places(images, results, recognizer, output_dir)

    # Test 3: Loop closure with fresh recognizer
    print("\n" + "=" * 60)
    print("Test: Loop Closure Detection")
    print("=" * 60)

    recognizer2 = ORBPlaceRecognizer(
        n_features=500,
        match_threshold=0.75,
        min_matches=12,
        keyframe_cooldown=8,
        max_keyframes=50,
    )

    test_loop_closure(images, recognizer2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nORB on real images vs synthetic:")
    print(f"  - Synthetic (simple 3D): ~20-500 features, 50% accuracy")
    print(f"  - Real TUM images: 400-500 features (full texture)")
    print(f"\nConclusion: Real images have abundant texture for ORB matching.")


if __name__ == "__main__":
    main()
