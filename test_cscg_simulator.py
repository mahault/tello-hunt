"""
Test CSCG mapping using the 3D simulator.

This lets you test the mapping system without a drone.
Walk through the virtual house and watch the spatial map build.

Controls:
    W/Up    - Move forward
    S/Down  - Move backward
    A/Left  - Turn left
    D/Right - Turn right
    R       - Reset map
    Q/ESC   - Quit
"""

import cv2
import numpy as np
import time

# Simulator
from simulator import Simple3DSimulator

# CSCG modules
from mapping.cscg import CSCGWorldModel
from utils.spatial_map import SpatialMap, combine_with_map


def main():
    print("=" * 60)
    print("CSCG Mapping Test with 3D Simulator")
    print("=" * 60)
    print()
    print("Controls:")
    print("  W/Up    - Move forward")
    print("  S/Down  - Move backward")
    print("  A/Left  - Turn left")
    print("  D/Right - Turn right")
    print("  R       - Reset map")
    print("  Q/ESC   - Quit")
    print()
    print("Watch the spatial map (right panel) build as you explore!")
    print("=" * 60)

    # Initialize simulator
    sim = Simple3DSimulator(width=640, height=480)

    # Initialize CSCG
    print("\nInitializing CSCG...")
    cscg = CSCGWorldModel(
        n_clones_per_token=8,
        n_tokens=32,
        embedding_dim=512,
        use_hybrid_tokenizer=False,  # Use pure CLIP embeddings
    )

    # Initialize spatial map
    spatial_map = SpatialMap(width=400, height=400)

    # Pre-load CLIP
    print("Loading CLIP model (this may take a moment)...")
    frame = sim.render()
    encoder = cscg._get_image_encoder()
    encoder.encode(frame)  # Warm up
    print("CLIP ready!")

    # Create window
    cv2.namedWindow("CSCG Simulator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CSCG Simulator", 1200, 500)

    frame_count = 0
    last_action = 0

    print("\nStarting simulation. Use WASD or arrow keys to move.")

    while True:
        # Render simulator view
        frame = sim.render()
        frame_count += 1

        # Run CSCG localization
        loc_result = cscg.localize(
            frame=frame,
            action_taken=last_action,
            observation_token=None,
        )

        # Update spatial map
        spatial_map.update(loc_result.token, last_action)

        # Reset action (we only record action when key is pressed)
        last_action = 0

        # Draw CSCG info on frame
        info_y = 60
        cv2.putText(frame, f"Token: {loc_result.token}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Clone: {loc_result.clone_state}", (10, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {loc_result.confidence:.1%}", (10, info_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"VFE: {loc_result.vfe:.2f}", (10, info_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if loc_result.new_token_discovered:
            cv2.putText(frame, "NEW TOKEN!", (10, info_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Render spatial map
        map_img = spatial_map.render()

        # Combine views
        display = combine_with_map(frame, map_img, target_height=480)

        cv2.imshow("CSCG Simulator", display)

        # Log periodically
        if frame_count % 30 == 0:
            pos = sim.get_position()
            print(f"[{frame_count}] Pos=({pos[0]:.1f}, {pos[1]:.1f}) "
                  f"Token={loc_result.token} Clone={loc_result.clone_state} "
                  f"Tokens={cscg.n_locations} VFE={loc_result.vfe:.2f}")

        # Handle input
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            print("\nQuitting...")
            break

        elif key in (ord('w'), ord('W'), 82):  # W or Up
            sim.move(1)
            last_action = 1

        elif key in (ord('s'), ord('S'), 84):  # S or Down
            sim.move(2)
            last_action = 2

        elif key in (ord('a'), ord('A'), 81):  # A or Left
            sim.move(3)
            last_action = 3

        elif key in (ord('d'), ord('D'), 83):  # D or Right
            sim.move(4)
            last_action = 4

        elif key in (ord('r'), ord('R')):  # Reset
            print("\nResetting map...")
            cscg.reset_belief()
            spatial_map.reset()
            sim.x, sim.y, sim.angle = 3.5, 2.0, np.pi / 2
            print("Map reset. Starting fresh.")

    cv2.destroyAllWindows()

    # Print final stats
    print("\n" + "=" * 60)
    print("Final Statistics:")
    print(f"  Tokens discovered: {cscg.n_locations}")
    print(f"  Clone states: {cscg.n_clone_states}")
    print(f"  Places in spatial map: {len(spatial_map._positions)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
