"""
Full Pipeline Test with 3D Simulator.

Tests the entire POMDP pipeline with two world model options:

1. CSCG World Model - Clone-structured cognitive graph with ORB
2. Semantic World Model - EFE-based exploration with room/object priors

The simulated drone explores autonomously, building a map.
You can also add simulated people to test hunting mode.

Controls:
    SPACE   - Toggle autonomous/manual control
    E       - Switch to Exploration mode
    H       - Switch to Hunting mode
    P       - Spawn/remove a person in current room
    M       - Print map summary (semantic mode)
    1       - Use CSCG world model
    2       - Use Semantic world model (EFE + priors)
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

# Semantic world model (EFE-based exploration)
try:
    from mapping.semantic_world_model import SemanticWorldModel, SemanticPrior
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
    print("Semantic world model not available")

# YOLO for object detection
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("YOLO not available (pip install ultralytics)")

# Monocular depth estimation (realistic - works on actual drone)
try:
    from utils.depth_estimator import DepthEstimator
    HAS_DEPTH = True
except ImportError:
    HAS_DEPTH = False
    DepthEstimator = None
    print("Depth estimation not available (pip install transformers)")

# Spatial mapping (occupancy grid + odometry)
from utils.occupancy_map import SpatialMapper

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


class SemanticWorldModelAdapter:
    """Adapter to make SemanticWorldModel look like WorldModel for exploration mode."""

    def __init__(self, semantic_model: 'SemanticWorldModel'):
        self._model = semantic_model

    @property
    def n_locations(self) -> int:
        # Use CSCG's place count (the ground truth for place recognition)
        return self._model.n_places

    def get_exploration_target(self):
        """Get EFE-based exploration action."""
        # Compute unified EFE (CSCG epistemic + semantic pragmatic)
        efe, info = self._model.compute_efe()
        # Convert EFE to probabilities (lower EFE = higher prob)
        action_scores = -efe  # Negate so lower EFE = higher score
        action_scores = action_scores - action_scores.max()
        probs = np.exp(action_scores)
        probs /= probs.sum()
        best_action = int(np.argmin(efe))
        return best_action, probs

    def get_exploration_urgency(self):
        """Get exploration urgency based on semantic coverage."""
        return self._model.get_exploration_urgency()

    def should_explore(self) -> bool:
        """Check if we should continue exploring."""
        urgency, _ = self._model.get_exploration_urgency()
        # Also check CSCG's urgency
        cscg_urgency, _ = self._model.cscg.get_exploration_urgency()
        return urgency > 0.3 or cscg_urgency > 0.3


class FullPipelineSimulator:
    """
    Full pipeline test combining simulator with POMDP controller.

    Supports two world model backends:
    - CSCG: Clone-structured cognitive graph with ORB place recognition
    - Semantic: EFE-based exploration with room/object priors
    """

    def __init__(self, use_glb: bool = True, use_semantic: bool = False):
        # 3D Simulator - prefer GLB for realistic textures
        self.use_glb = use_glb and HAS_GLB
        if self.use_glb:
            print("Using GLB simulator (realistic textures for ORB)")
            self.sim = GLBSimulator(width=640, height=480)
        else:
            print("Using Simple 3D simulator")
            self.sim = Simple3DSimulator(width=640, height=480)

        # World model selection
        self.use_semantic = use_semantic and HAS_SEMANTIC

        if self.use_semantic:
            # Semantic World Model (EFE-based exploration)
            print("Initializing Semantic World Model (EFE + priors)...")
            # Create CSCG first (bottom layer)
            print("Initializing CSCG World Model with ORB (for semantic layer)...")
            self.cscg = CSCGWorldModel(
                n_clones_per_token=8,
                n_tokens=32,
                encoder_type="orb",
            )
            # Create semantic priors (matching the GLB house model - dynamic layout)
            self.semantic_prior = SemanticPrior(
                room_types=["living_room", "kitchen", "bedroom", "bathroom", "hallway"],
                objects_per_room={
                    "living_room": ["couch", "tv", "chair"],
                    "kitchen": ["refrigerator", "oven", "sink", "microwave", "dining table"],
                    "bedroom": ["bed", "chair"],
                    "bathroom": ["toilet", "sink"],
                    "hallway": [],
                },
            )
            # SemanticWorldModel wraps CSCG (top layer)
            self.semantic_model = SemanticWorldModel(
                cscg=self.cscg,
                prior=self.semantic_prior,
                pragmatic_weight=0.4,  # Balance epistemic/pragmatic
            )
            self.world_model_adapter = SemanticWorldModelAdapter(self.semantic_model)
        else:
            # CSCG World Model with ORB (default - 87.5% accuracy)
            print("Initializing CSCG World Model with ORB...")
            self.cscg = CSCGWorldModel(
                n_clones_per_token=8,
                n_tokens=32,
                encoder_type="orb",  # ORB for place recognition
            )
            self.semantic_model = None
            self.world_model_adapter = CSCGWorldModelAdapter(self.cscg)

        # Spatial map (old topological visualization)
        self.spatial_map = SpatialMap(width=350, height=350)

        # Occupancy grid mapper (builds actual spatial map from depth + odometry)
        self.spatial_mapper = SpatialMapper(
            map_size=200,
            resolution=0.1,  # 10cm per cell
            use_visual_odometry=False,  # Use action-based (more reliable indoors)
        )

        # YOLO for object detection (used for room type inference)
        self.yolo = None
        if HAS_YOLO and self.use_semantic:
            print("Loading YOLO for object detection...")
            self.yolo = YOLO("yolov8n.pt")
            # Warm up
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.yolo(dummy, verbose=False)
            print("YOLO ready!")

        # Monocular depth estimation (realistic - same as real drone)
        self.depth_estimator = None
        self.use_depth = HAS_DEPTH and self.use_semantic
        if self.use_depth:
            print("Loading depth estimator (Depth Anything)...")
            self.depth_estimator = DepthEstimator(model_type="depth_anything")
            print("Depth estimator ready!")

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

        # Room classification tracking for analysis
        # Maps place_id -> {'ground_truth': [rooms], 'predicted': [rooms]}
        self._room_tracking: Dict[int, Dict[str, List[str]]] = {}

        model_type = "Semantic (EFE)" if self.use_semantic else "CSCG (ORB)"
        print(f"Ready! ({model_type} world model)")

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

        # Depth estimation (realistic - same method would work on real drone)
        depth_map = None
        depth_distances = None
        if self.depth_estimator is not None:
            depth_map = self.depth_estimator.estimate(frame)
            depth_distances = self.depth_estimator.get_obstacle_distances(depth_map)

        # Check for visible person
        visible_person = self._check_person_visibility()

        # Create observation
        obs = self._create_fake_observation(visible_person)

        # World model localization (CSCG or Semantic)
        if self.use_semantic:
            # Run YOLO detection for object-based room inference
            yolo_results = None
            if self.yolo is not None:
                yolo_results = self.yolo(frame, verbose=False)

            # Semantic world model
            sem_result = self.semantic_model.update(
                frame=frame,
                action_taken=self._last_action,
                yolo_detections=yolo_results,
                debug=(self._frame_count % 60 == 0),
            )
            # Create compatible loc_result
            loc_result = type('LocResult', (), {
                'token': sem_result['place_id'],
                'clone_state': sem_result['place_id'],
                'token_similarity': sem_result['confidence'],
                'new_token_discovered': sem_result['is_new_place'],
                'vfe': sem_result['vfe'],
                'belief': np.array([1.0]),
            })()

            # Update spatial map
            self.spatial_map.update(sem_result['place_id'], self._last_action)

            # Store semantic info for display
            self._semantic_result = sem_result

            # Track room classification for analysis
            place_id = sem_result['place_id']
            ground_truth = self.sim._get_current_room() if hasattr(self.sim, '_get_current_room') else "Unknown"
            predicted = sem_result.get('room_type') or "unknown"
            self._track_room(place_id, ground_truth, predicted)
        else:
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

            self._semantic_result = None

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

        # Depth-based obstacle prediction (realistic - predicts before moving)
        depth_blocked = False
        if depth_distances is not None and action in (1, 2):  # forward or backward
            direction = 'forward' if action == 1 else 'forward'  # backward uses same check
            path_clear, path_distance = self.depth_estimator.is_path_clear(
                depth_map, direction='forward', threshold=0.25
            )
            if not path_clear and action == 1:  # Only block forward based on depth
                depth_blocked = True
                if self._frame_count % 30 == 0:
                    print(f"  [DEPTH] Obstacle detected! Distance={path_distance:.2f}")

        # Execute action with debug
        debug_this_frame = (self._frame_count % 30 == 0)  # Debug every 30 frames
        moved = True
        if action > 0:
            # If depth predicts obstacle, still try but report to CSCG preemptively
            if depth_blocked:
                self.cscg.record_blocked_action(action)
                if self.mode == "exploration":
                    self.exploration.record_movement_result(action, False)

            moved = self.sim.move(action, debug=debug_this_frame)
            # Report movement result to exploration (for wall detection)
            if self.mode == "exploration" and not depth_blocked:
                self.exploration.record_movement_result(action, moved)
            # Report blocked actions to CSCG so it learns obstacles
            if not moved and not depth_blocked:
                self.cscg.record_blocked_action(action)
                if debug_this_frame:
                    print(f"  [PIPELINE] Action {action} blocked at place {self.cscg._prev_token}")

        self._last_action = action
        self._frame_count += 1

        # Update spatial mapper (builds occupancy grid from depth + odometry)
        place_id = loc_result.token if hasattr(loc_result, 'token') else None

        # Get semantic label (room type) for the current place
        place_label = None
        if self.use_semantic and self._semantic_result:
            place_label = self._semantic_result.get('room_type')
        elif self.use_glb:
            # Use ground truth room from simulator
            place_label = self.sim._get_current_room()

        self.spatial_mapper.update(
            frame=frame,
            action=action,
            moved=moved,
            depth_map=depth_map,
            place_id=place_id,
            place_label=place_label,
        )

        # Add common info
        result['frame'] = frame
        result['localization'] = loc_result
        result['mode'] = self.mode
        result['autonomous'] = self.autonomous
        result['action'] = action
        result['visible_person'] = visible_person is not None
        result['depth_map'] = depth_map
        result['depth_distances'] = depth_distances
        result['depth_blocked'] = depth_blocked
        result['mapper_stats'] = self.spatial_mapper.get_stats()

        return result

    def _exploration_update(self, obs: ObservationToken, loc_result) -> Dict:
        """Run exploration mode update."""
        # Use adapter so exploration sees CSCG's locations
        explore_result = self.exploration.update(obs, self.world_model_adapter, loc_result)

        # Periodically run CHMM learning to update transition structure
        if self._frame_count > 0 and self._frame_count % 100 == 0:
            self.cscg.learn(n_iter=3)

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
            # Reset GLB simulator position (use model center at door height)
            bounds = self.sim.trimesh_scene.bounds
            floor_y = bounds[0][1]
            ceiling_y = bounds[1][1]
            room_height = ceiling_y - floor_y
            door_midpoint = floor_y + room_height * 0.4

            self.sim.x = self.sim.model_center[0]
            self.sim.y = door_midpoint
            self.sim.z = self.sim.model_center[2]
            self.sim.yaw = 0.0
            self.sim.pitch = 0.0
        else:
            # Reset Simple 3D simulator position
            self.sim.x, self.sim.y, self.sim.angle = 3.5, 2.0, np.pi / 2

        # Reset world model
        if self.use_semantic:
            self.semantic_model.reset()
        else:
            self.cscg.reset_belief()
            if self.cscg.use_orb:
                self.cscg._orb_recognizer.reset()

        self.spatial_map.reset()
        self.spatial_mapper.reset()
        self.exploration.reset()
        self.interaction.reset_to_searching()
        self.people.clear()
        self.mode = "exploration"
        self._last_action = 0
        self._frame_count = 0
        self._room_tracking.clear()
        print("Reset complete")

    def _track_room(self, place_id: int, ground_truth: str, predicted: str):
        """Track room classification for later analysis."""
        if place_id not in self._room_tracking:
            self._room_tracking[place_id] = {'ground_truth': [], 'predicted': []}
        self._room_tracking[place_id]['ground_truth'].append(ground_truth)
        self._room_tracking[place_id]['predicted'].append(predicted)

    def get_room_analysis(self) -> str:
        """Generate room classification accuracy analysis."""
        if not self._room_tracking:
            return "No room tracking data available."

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("ROOM CLASSIFICATION ANALYSIS")
        lines.append("=" * 70)

        # Normalize room names for comparison
        def normalize(room: str) -> str:
            room = room.lower().replace(" ", "_").replace("-", "_")
            # Map variations
            mappings = {
                "living_room": "living_room", "livingroom": "living_room",
                "bedroom": "bedroom", "bed_room": "bedroom",
                "bathroom": "bathroom", "bath_room": "bathroom",
                "kitchen": "kitchen",
                "hallway": "hallway", "hall": "hallway",
                "unknown": "unknown", "?": "unknown",
            }
            return mappings.get(room, room)

        # Per-place analysis
        lines.append("\nPer-Place Classification:")
        lines.append("-" * 50)

        total_correct = 0
        total_observations = 0
        place_results = []

        for place_id in sorted(self._room_tracking.keys()):
            data = self._room_tracking[place_id]
            gt_list = [normalize(r) for r in data['ground_truth']]
            pred_list = [normalize(r) for r in data['predicted']]

            # Most common ground truth and prediction for this place
            from collections import Counter
            gt_counts = Counter(gt_list)
            pred_counts = Counter(pred_list)

            gt_majority = gt_counts.most_common(1)[0][0] if gt_counts else "unknown"
            pred_majority = pred_counts.most_common(1)[0][0] if pred_counts else "unknown"

            # Count frame-level matches (excluding unknown)
            matches = sum(1 for gt, pred in zip(gt_list, pred_list)
                         if gt != "unknown" and pred != "unknown" and gt == pred)
            valid_obs = sum(1 for gt, pred in zip(gt_list, pred_list)
                           if gt != "unknown" and pred != "unknown")

            accuracy = matches / valid_obs * 100 if valid_obs > 0 else 0
            total_correct += matches
            total_observations += valid_obs

            # Determine if place classification is correct (majority vote)
            is_correct = gt_majority == pred_majority and gt_majority != "unknown"
            status = "OK" if is_correct else ("?" if gt_majority == "unknown" or pred_majority == "unknown" else "WRONG")

            place_results.append({
                'place_id': place_id,
                'gt': gt_majority,
                'pred': pred_majority,
                'accuracy': accuracy,
                'n_obs': len(gt_list),
                'status': status,
            })

            lines.append(f"  Place_{place_id:2d}: GT={gt_majority:12s} Pred={pred_majority:12s} "
                        f"({len(gt_list):3d} obs, {accuracy:5.1f}% match) [{status}]")

        # Summary
        lines.append("\n" + "-" * 50)
        lines.append("Summary:")

        overall_acc = total_correct / total_observations * 100 if total_observations > 0 else 0
        lines.append(f"  Frame-level accuracy: {total_correct}/{total_observations} = {overall_acc:.1f}%")

        # Room coverage
        gt_rooms = set(r['gt'] for r in place_results if r['gt'] != 'unknown')
        pred_rooms = set(r['pred'] for r in place_results if r['pred'] != 'unknown')
        lines.append(f"  Ground truth rooms visited: {', '.join(sorted(gt_rooms)) or 'none'}")
        lines.append(f"  Predicted rooms found: {', '.join(sorted(pred_rooms)) or 'none'}")

        # Correct/wrong counts
        correct_places = sum(1 for r in place_results if r['status'] == 'OK')
        wrong_places = sum(1 for r in place_results if r['status'] == 'WRONG')
        unknown_places = sum(1 for r in place_results if r['status'] == '?')
        lines.append(f"  Place classification: {correct_places} correct, {wrong_places} wrong, {unknown_places} unknown")

        # Confusion matrix (simplified)
        lines.append("\nConfusion (GT -> Pred mismatches):")
        confusions = {}
        for r in place_results:
            if r['status'] == 'WRONG':
                key = f"{r['gt']} -> {r['pred']}"
                confusions[key] = confusions.get(key, 0) + 1
        if confusions:
            for conf, count in sorted(confusions.items(), key=lambda x: -x[1]):
                lines.append(f"  {conf}: {count} place(s)")
        else:
            lines.append("  (no misclassifications)")

        lines.append("=" * 70)
        return "\n".join(lines)

    def render_display(self, result: Dict) -> np.ndarray:
        """Render the full display with all panels."""
        frame = result['frame'].copy()
        loc = result['localization']

        # Draw mode indicator
        mode_color = (0, 255, 255) if self.mode == "exploration" else (0, 255, 0)
        auto_text = "AUTO" if self.autonomous else "MANUAL"
        model_text = "Semantic" if self.use_semantic else "CSCG"
        cv2.putText(frame, f"{self.mode.upper()} ({auto_text}) [{model_text}]", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        y = 55

        if self.use_semantic and self._semantic_result:
            # Draw Semantic world model info
            sem = self._semantic_result
            cv2.putText(frame, f"Place: {sem['place_name']} (id={sem['place_id']})", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            room_type = sem.get('room_type') or 'unknown'
            cv2.putText(frame, f"Room: {room_type}", (10, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(frame, f"Places: {sem['n_places']} | Conf: {sem['confidence']:.2f}", (10, y+40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show room coverage
            urgency, reason = self.semantic_model.get_exploration_urgency()
            cv2.putText(frame, f"Urgency: {urgency:.2f}", (10, y+60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"{reason}", (10, y+80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Show found rooms
            room_cov = self.semantic_model._compute_room_coverage()
            found = ', '.join(room_cov.keys()) or 'none'
            cv2.putText(frame, f"Found: {found}", (10, y+100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Objects at current place
            objects = sem.get('objects_here', [])
            if objects:
                obj_str = ', '.join(objects[:4])
                cv2.putText(frame, f"Objects: {obj_str}", (10, y+120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        else:
            # Draw CSCG info
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

        # Depth-based obstacle warning
        depth_distances = result.get('depth_distances')
        if depth_distances:
            fwd_dist = depth_distances.get('forward_path', 1.0)
            # Color based on distance
            if fwd_dist < 0.3:
                color = (0, 0, 255)  # Red
                status = "BLOCKED"
            elif fwd_dist < 0.5:
                color = (0, 165, 255)  # Orange
                status = "CAUTION"
            else:
                color = (0, 255, 0)  # Green
                status = "CLEAR"
            cv2.putText(frame, f"Depth: {status} ({fwd_dist:.2f})", (frame.shape[1] - 180, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if result.get('depth_blocked'):
            cv2.putText(frame, "OBSTACLE!", (frame.shape[1]//2 - 50, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show mapper stats (pose and exploration %)
        mapper_stats = result.get('mapper_stats', {})
        if mapper_stats:
            pose_x = mapper_stats.get('pose_x', 0)
            pose_y = mapper_stats.get('pose_y', 0)
            pose_yaw = mapper_stats.get('pose_yaw_deg', 0)
            explored = mapper_stats.get('explored_pct', 0)
            cv2.putText(frame, f"Pos: ({pose_x:.1f}, {pose_y:.1f}) {pose_yaw:.0f}deg",
                       (10, frame.shape[0] - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"Mapped: {explored:.1f}%",
                       (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Render occupancy map (actual spatial map being built)
        occupancy_map_img = self.spatial_mapper.render(size=200)

        # Combine with depth visualization if available
        depth_map = result.get('depth_map')
        if depth_map is not None and self.depth_estimator is not None:
            # Create small depth thumbnail
            depth_vis = self.depth_estimator.visualize_depth(depth_map)
            depth_small = cv2.resize(depth_vis, (120, 90))
            # Put in bottom-left corner of frame
            frame[-95:-5, 5:125] = depth_small

        # Create side panel with occupancy map
        h = frame.shape[0]
        panel = np.zeros((h, 210, 3), dtype=np.uint8)
        panel.fill(30)  # Dark gray background

        # Place occupancy map in panel
        map_y = (h - 200) // 2
        panel[map_y:map_y+200, 5:205] = occupancy_map_img

        # Combine frame with panel
        display = np.hstack([frame, panel])

        return display


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Full Pipeline Test')
    parser.add_argument('--semantic', action='store_true', help='Use Semantic world model (EFE + priors)')
    args = parser.parse_args()

    print("=" * 70)
    print("FULL PIPELINE TEST - Autonomous Exploration")
    print("=" * 70)
    print()
    print("World Models:")
    print("  1 - CSCG: Clone-structured cognitive graph with ORB")
    print("  2 - Semantic: EFE-based exploration with room/object priors")
    print()
    print("Controls:")
    print("  SPACE   - Toggle autonomous/manual control")
    print("  E       - Switch to Exploration mode")
    print("  H       - Switch to Hunting mode")
    print("  M       - Print map summary (semantic mode)")
    print("  P       - Spawn a person in current room")
    print("  R       - Reset everything")
    print("  Q/ESC   - Quit")
    print()
    print("In MANUAL mode: WASD or Arrow keys to move")
    print("=" * 70)
    print()

    # Initialize with selected model
    use_semantic = args.semantic or False
    pipeline = FullPipelineSimulator(use_semantic=use_semantic)

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

            if pipeline.use_semantic:
                sem = pipeline._semantic_result
                place_name = sem['place_name'] if sem else 'Unknown'
                room_type = sem.get('room_type') or '?' if sem else '?'
                n_places = sem['n_places'] if sem else 0
                urgency, _ = pipeline.semantic_model.get_exploration_urgency()

                print(f"[{frame_count:4d}] {place_name:12s} room={room_type:12s} "
                      f"places={n_places} urgency={urgency:.2f}")
            else:
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

        elif key in (ord('m'), ord('M')):
            if pipeline.use_semantic:
                print("\n" + pipeline.semantic_model.get_map_summary())
            else:
                print("\nMap summary only available in semantic mode")

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
    print(f"  Final mode: {pipeline.mode}")

    # Spatial mapping stats
    mapper_stats = pipeline.spatial_mapper.get_stats()
    print(f"\n  Spatial Mapping:")
    print(f"    Final position: ({mapper_stats['pose_x']:.2f}, {mapper_stats['pose_y']:.2f}) m")
    print(f"    Final heading: {mapper_stats['pose_yaw_deg']:.1f} deg")
    print(f"    Map explored: {mapper_stats['explored_pct']:.1f}%")
    print(f"    Places on map: {mapper_stats['places_mapped']}")
    print(f"    Trajectory points: {mapper_stats['trajectory_length']}")

    # Show discovered places (ORB keyframes)
    if pipeline.cscg.use_orb:
        print("\n  Discovered places (ORB keyframes):")
        for kf in pipeline.cscg._orb_recognizer.keyframes:
            print(f"    - {kf.name} (visited {kf.visit_count}x)")

    print("=" * 70)

    # Room classification analysis (semantic mode only)
    if pipeline.use_semantic:
        print(pipeline.get_room_analysis())


if __name__ == "__main__":
    main()
