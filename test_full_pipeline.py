"""
Full Pipeline Test with 3D Simulator.

Tests the entire POMDP pipeline:
- CSCG World Model with ORB place recognition (87.5% accuracy)
- Exploration Mode (autonomous movement)
- Hunting Mode (person search)
- Spatial map visualization

The simulated drone explores autonomously, building a map.
You can also add simulated people to test hunting mode.

Controls:
    SPACE   - Toggle autonomous/manual control
    E       - Switch to Exploration mode
    H       - Switch to Hunting mode
    P       - Spawn/remove a person in current room
    G       - Toggle between GLB and Simple simulators
    R       - Reset everything
    Q/ESC   - Quit

In manual mode:
    W/Up    - Move forward
    S/Down  - Move backward
    A/Left  - Turn left
    D/Right - Turn right
"""

import cv2
import numpy as np
import time
import os

# Simulators
from simulator import Simple3DSimulator
try:
    from simulator.glb_simulator import GLBSimulator
    HAS_GLB = True
except ImportError:
    HAS_GLB = False
    print("GLB simulator not available, using simple 3D only")

# POMDP modules
from pomdp import (
    WorldModel,
    HumanSearchPOMDP,
    InteractionModePOMDP,
    ExplorationModePOMDP,
    encode_yolo_detections,
    exploration_action_to_rc_control,
    action_to_rc_control,
)
from pomdp.observation_encoder import ObservationToken
from pomdp.config import MAX_FB, MAX_YAW, SEARCH_YAW

# CSCG modules
from mapping.cscg import CSCGWorldModel
from utils.spatial_map import SpatialMap, combine_with_map

# For fake YOLO detections
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class FakeBox:
    """Fake YOLO detection box."""
    xyxy: List[np.ndarray]
    cls: List[int]
    conf: List[float]


class FakeBoxes:
    """Container for fake YOLO boxes."""
    def __init__(self, boxes: List[FakeBox] = None):
        self._boxes = boxes or []

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class SimulatedPerson:
    """A simulated person in the environment."""
    def __init__(self, room_name: str, x: float, y: float):
        self.room_name = room_name
        self.x = x
        self.y = y
        self.visible = False


class CSCGWorldModelAdapter:
    """Adapter to make CSCG look like WorldModel for exploration mode."""

    def __init__(self, cscg: CSCGWorldModel):
        self._cscg = cscg

    @property
    def n_locations(self) -> int:
        return self._cscg.n_locations

    def get_exploration_target(self):
        """Get exploration target from CSCG."""
        return self._cscg.get_exploration_target()

    def get_exploration_urgency(self):
        """Get exploration urgency from CSCG."""
        return self._cscg.get_exploration_urgency()

    def should_explore(self) -> bool:
        """Check if we should continue exploring."""
        return self._cscg.should_explore()


class FullPipelineSimulator:
    """
    Full pipeline test combining simulator with POMDP controller.
    """

    def __init__(self, use_glb: bool = True):
        # 3D Simulator - prefer GLB for realistic textures
        self.use_glb = use_glb and HAS_GLB
        if self.use_glb:
            print("Using GLB simulator (realistic textures for ORB)")
            self.sim = GLBSimulator(width=640, height=480)
        else:
            print("Using Simple 3D simulator")
            self.sim = Simple3DSimulator(width=640, height=480)

        # CSCG World Model with ORB (default - 87.5% accuracy)
        print("Initializing CSCG World Model with ORB...")
        self.cscg = CSCGWorldModel(
            n_clones_per_token=8,
            n_tokens=32,
            encoder_type="orb",  # ORB for place recognition (87.5% accuracy)
        )

        # Adapter to make CSCG look like WorldModel for exploration
        self.world_model_adapter = CSCGWorldModelAdapter(self.cscg)

        # Spatial map
        self.spatial_map = SpatialMap(width=350, height=350)

        # Other POMDPs
        self.exploration = ExplorationModePOMDP()
        self.human_search = HumanSearchPOMDP(n_locations=1)
        self.interaction = InteractionModePOMDP()

        # Mode
        self.mode = "exploration"
        self.autonomous = True

        # Simulated people
        self.people: List[SimulatedPerson] = []

        # Action tracking
        self._last_action = 0
        self._frame_count = 0

        print("Ready! (ORB place recognition enabled)")

    def add_person(self, room_name: str = None):
        """Add a simulated person."""
        if self.use_glb:
            print("Person simulation not supported in GLB mode")
            return

        if room_name is None:
            room_name = self.sim._get_current_room()

        # Find room center
        for room in self.sim.rooms:
            if room.name == room_name:
                person = SimulatedPerson(room_name, room.center_x, room.center_y)
                self.people.append(person)
                print(f"Added person in {room_name}")
                return

        print(f"Room '{room_name}' not found")

    def remove_people(self):
        """Remove all simulated people."""
        self.people.clear()
        print("Removed all people")

    def _check_person_visibility(self) -> Optional[SimulatedPerson]:
        """Check if any person is visible from current position."""
        if self.use_glb:
            return None  # No person simulation in GLB mode

        for person in self.people:
            # Simple visibility check: same room and facing roughly toward them
            current_room = self.sim._get_current_room()
            if person.room_name != current_room:
                continue

            # Check if facing toward person
            dx = person.x - self.sim.x
            dy = person.y - self.sim.y
            dist = np.sqrt(dx*dx + dy*dy)

            if dist > 5.0:  # Too far
                continue

            # Angle to person
            angle_to_person = np.arctan2(dy, dx)
            angle_diff = abs(angle_to_person - self.sim.angle)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)

            if angle_diff < np.pi / 3:  # Within ~60 degree FOV
                person.visible = True
                return person

        return None

    def _create_fake_observation(self, visible_person: Optional[SimulatedPerson]) -> ObservationToken:
        """Create a fake observation token."""
        n_types = 17  # Number of object types

        if visible_person:
            # Person detected - create fake detection
            return ObservationToken(
                object_levels=np.zeros(n_types, dtype=np.int32),
                object_counts=np.zeros(n_types, dtype=np.int32),
                object_max_conf=np.zeros(n_types, dtype=np.float32),
                object_avg_pos=np.zeros((n_types, 2), dtype=np.float32),
                object_avg_area=np.zeros(n_types, dtype=np.float32),
                person_detected=True,
                person_area=0.15,
                person_cx=0.0,  # Centered
                person_cy=0.0,
                person_conf=0.9,
                person_obs_idx=2,  # Center
            )
        else:
            # No person
            return ObservationToken(
                object_levels=np.zeros(n_types, dtype=np.int32),
                object_counts=np.zeros(n_types, dtype=np.int32),
                object_max_conf=np.zeros(n_types, dtype=np.float32),
                object_avg_pos=np.zeros((n_types, 2), dtype=np.float32),
                object_avg_area=np.zeros(n_types, dtype=np.float32),
                person_detected=False,
                person_area=0.0,
                person_cx=0.0,
                person_cy=0.0,
                person_conf=0.0,
                person_obs_idx=0,  # Not detected
            )

    def update(self, manual_action: int = 0) -> Dict:
        """
        Run one update cycle.

        Args:
            manual_action: Manual action if not autonomous (0=none, 1=fwd, 2=back, 3=left, 4=right)

        Returns:
            Result dict with all info
        """
        # Render frame
        frame = self.sim.render()

        # Check for visible person
        visible_person = self._check_person_visibility()

        # Create observation
        obs = self._create_fake_observation(visible_person)

        # CSCG localization
        loc_result = self.cscg.localize(
            frame=frame,
            action_taken=self._last_action,
            observation_token=obs,
        )

        # Update spatial map
        self.spatial_map.update(loc_result.token, self._last_action)

        # Expand human search if needed
        if loc_result.new_token_discovered:
            n_tokens = self.cscg.n_locations
            if n_tokens > self.human_search.n_locations:
                self.human_search.expand_to_locations(n_tokens)

        # Mode-specific update
        if self.mode == "exploration":
            result = self._exploration_update(obs, loc_result)
        else:
            result = self._hunting_update(obs, loc_result, visible_person)

        # Determine action
        if self.autonomous:
            action = result['action_idx']
        else:
            action = manual_action

        # Execute action with debug
        debug_this_frame = (self._frame_count % 30 == 0)  # Debug every 30 frames
        moved = True
        if action > 0:
            moved = self.sim.move(action, debug=debug_this_frame)
            # Report movement result to exploration (for wall detection)
            if self.mode == "exploration":
                self.exploration.record_movement_result(action, moved)
            if debug_this_frame and action == 1 and not moved:
                print(f"  [PIPELINE] Forward blocked at ({self.sim.x:.1f},{self.sim.y:.1f})")

        self._last_action = action
        self._frame_count += 1

        # Add common info
        result['frame'] = frame
        result['localization'] = loc_result
        result['mode'] = self.mode
        result['autonomous'] = self.autonomous
        result['action'] = action
        result['visible_person'] = visible_person is not None

        return result

    def _exploration_update(self, obs: ObservationToken, loc_result) -> Dict:
        """Run exploration mode update."""
        # Use adapter so exploration sees CSCG's locations
        explore_result = self.exploration.update(obs, self.world_model_adapter, loc_result)

        # Debug: print state transition info periodically
        if self.exploration._total_frames % 100 == 1:
            print(f"  [DEBUG] rotations={self.exploration._rotations_completed}, "
                  f"frames_in_state={self.exploration._frames_in_state}, "
                  f"n_locations={self.world_model_adapter.n_locations}")

        # Check for transition to hunting
        if explore_result.should_transition_to_hunt:
            print(f"\nTransitioning to HUNTING: {explore_result.transition_reason}")
            self.mode = "hunting"

        return {
            'action_idx': explore_result.selected_action,
            'action_name': explore_result.selected_action_name,
            'exploration_state': explore_result.exploration_state,
            'vfe': explore_result.current_vfe,
            'mean_vfe': explore_result.mean_vfe,
        }

    def _hunting_update(self, obs: ObservationToken, loc_result, visible_person) -> Dict:
        """Run hunting mode update."""
        # Update human search
        human_result = self.human_search.update(obs, drone_location=loc_result.token)

        # Update interaction
        interaction_result = self.interaction.update(obs)

        return {
            'action_idx': interaction_result.selected_action_idx,
            'action_name': interaction_result.selected_action,
            'engagement_state': interaction_result.engagement_state,
            'person_detected': visible_person is not None,
            'search_target': human_result.search_target,
        }

    def reset(self):
        """Reset everything."""
        if self.use_glb:
            # Reset GLB simulator position
            self.sim.x = self.sim.model_center[0]
            self.sim.y = 200.0
            self.sim.z = self.sim.model_center[2]
            self.sim.yaw = 0.0
        else:
            # Reset Simple 3D simulator position
            self.sim.x, self.sim.y, self.sim.angle = 3.5, 2.0, np.pi / 2

        self.cscg.reset_belief()
        # Reset ORB recognizer if using ORB
        if self.cscg.use_orb:
            self.cscg._orb_recognizer.reset()
        self.spatial_map.reset()
        self.exploration.reset()
        self.interaction.reset_to_searching()
        self.people.clear()
        self.mode = "exploration"
        self._last_action = 0
        print("Reset complete")

    def render_display(self, result: Dict) -> np.ndarray:
        """Render the full display with all panels."""
        frame = result['frame'].copy()
        loc = result['localization']

        # Draw mode indicator
        mode_color = (0, 255, 255) if self.mode == "exploration" else (0, 255, 0)
        auto_text = "AUTO" if self.autonomous else "MANUAL"
        cv2.putText(frame, f"{self.mode.upper()} ({auto_text})", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # Draw CSCG info
        y = 55
        place_name = self.cscg.get_place_name(loc.token)
        cv2.putText(frame, f"Place: {place_name} (id={loc.token})", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Clone: {loc.clone_state}/{self.cscg.n_clone_states}", (10, y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Places: {self.cscg.n_locations} | Conf: {loc.token_similarity:.2f}", (10, y+40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mode-specific info
        if self.mode == "exploration":
            cv2.putText(frame, f"State: {result.get('exploration_state', '?')}", (10, y+60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"VFE: {result.get('mean_vfe', 0):.2f}", (10, y+80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, f"State: {result.get('engagement_state', '?')}", (10, y+60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if result.get('person_detected'):
                cv2.putText(frame, "PERSON DETECTED!", (10, y+80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw action
        action_name = result.get('action_name', 'stay')
        cv2.putText(frame, f"Action: {action_name}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # New token indicator
        if loc.new_token_discovered:
            cv2.putText(frame, "NEW PLACE!", (frame.shape[1]//2 - 60, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw people indicator
        if self.people:
            cv2.putText(frame, f"People: {len(self.people)}", (frame.shape[1] - 100, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        # Render spatial map
        map_img = self.spatial_map.render()

        # Combine
        display = combine_with_map(frame, map_img, target_height=480)

        return display


def main():
    print("=" * 70)
    print("FULL PIPELINE TEST - Autonomous Exploration with CSCG")
    print("=" * 70)
    print()
    print("Controls:")
    print("  SPACE   - Toggle autonomous/manual control")
    print("  E       - Switch to Exploration mode")
    print("  H       - Switch to Hunting mode")
    print("  P       - Spawn a person in current room")
    print("  R       - Reset everything")
    print("  Q/ESC   - Quit")
    print()
    print("In MANUAL mode: WASD or Arrow keys to move")
    print("=" * 70)
    print()

    # Initialize
    pipeline = FullPipelineSimulator()

    # Create window
    cv2.namedWindow("Full Pipeline Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Full Pipeline Test", 1200, 520)

    frame_count = 0
    manual_action = 0

    print("\nStarting autonomous exploration...")
    print("The drone will explore the house on its own.")
    print("Press SPACE to take manual control.\n")

    while True:
        frame_count += 1

        # Run update
        result = pipeline.update(manual_action=manual_action)
        manual_action = 0  # Reset manual action

        # Render display
        display = pipeline.render_display(result)
        cv2.imshow("Full Pipeline Test", display)

        # Log periodically
        if frame_count % 60 == 0:
            loc = result['localization']
            place_name = pipeline.cscg.get_place_name(loc.token)

            if pipeline.mode == "exploration":
                print(f"[{frame_count:4d}] {place_name:12s} | Places={pipeline.cscg.n_locations} "
                      f"Clones={pipeline.cscg.n_clone_states} "
                      f"State={result.get('exploration_state', '?')} VFE={result.get('mean_vfe', 0):.2f}")
            else:
                print(f"[{frame_count:4d}] {place_name:12s} | "
                      f"State={result.get('engagement_state', '?')} "
                      f"Person={'YES' if result.get('person_detected') else 'no'}")

        # Handle input
        key = cv2.waitKey(50) & 0xFF  # Slower for autonomous mode

        if key in (ord('q'), ord('Q'), 27):
            print("\nQuitting...")
            break

        elif key == ord(' '):  # Space - toggle autonomous
            pipeline.autonomous = not pipeline.autonomous
            mode_str = "AUTONOMOUS" if pipeline.autonomous else "MANUAL"
            print(f"\nSwitched to {mode_str} control")

        elif key in (ord('e'), ord('E')):
            pipeline.mode = "exploration"
            pipeline.exploration.reset()
            print("\nSwitched to EXPLORATION mode")

        elif key in (ord('h'), ord('H')):
            pipeline.mode = "hunting"
            pipeline.interaction.reset_to_searching()
            print("\nSwitched to HUNTING mode")

        elif key in (ord('p'), ord('P')):
            if pipeline.people:
                pipeline.remove_people()
            else:
                pipeline.add_person()

        elif key in (ord('r'), ord('R')):
            pipeline.reset()

        # Manual controls
        elif not pipeline.autonomous:
            if key in (ord('w'), ord('W'), 82):
                manual_action = 1
            elif key in (ord('s'), ord('S'), 84):
                manual_action = 2
            elif key in (ord('a'), ord('A'), 81):
                manual_action = 3
            elif key in (ord('d'), ord('D'), 83):
                manual_action = 4

    cv2.destroyAllWindows()

    # Cleanup GLB simulator
    if pipeline.use_glb:
        pipeline.sim.close()

    # Final stats
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"  Frames processed: {frame_count}")
    print(f"  Places discovered: {pipeline.cscg.n_locations}")
    print(f"  Clone states: {pipeline.cscg.n_clone_states}")
    print(f"  Places mapped: {len(pipeline.spatial_map._positions)}")
    print(f"  Final mode: {pipeline.mode}")

    # Show discovered places (ORB keyframes)
    if pipeline.cscg.use_orb:
        print("\n  Discovered places (ORB keyframes):")
        for kf in pipeline.cscg._orb_recognizer.keyframes:
            print(f"    - {kf.name} (visited {kf.visit_count}x)")

    print("=" * 70)


if __name__ == "__main__":
    main()
