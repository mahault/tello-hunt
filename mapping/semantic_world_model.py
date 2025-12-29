"""
Semantic World Model with Active Inference.

ARCHITECTURE:
┌─────────────────────────────────────────────┐
│  SEMANTIC LAYER (THIS FILE) - Top           │
│  • Room type priors & classification        │
│  • Object tracking per place                │
│  • EFE pragmatic term (find all rooms)      │
├─────────────────────────────────────────────┤
│  CSCG LAYER - Middle (imported)             │
│  • ORB place recognition                    │
│  • Clone state disambiguation               │
│  • Transition learning                      │
│  • EFE epistemic term                       │
├─────────────────────────────────────────────┤
│  PERCEPTION - Bottom                        │
│  • ORB keypoints, YOLO detections           │
└─────────────────────────────────────────────┘

The agent has PREFERENCES (priors) about what it expects to find:
- Expected room types (kitchen, bedroom, bathroom, living_room)
- Expected objects per room type

Exploration minimizes Expected Free Energy:
- Epistemic (from CSCG): Reduce uncertainty about transitions
- Pragmatic (from priors): Find all expected rooms and objects
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING
from pathlib import Path
import cv2

if TYPE_CHECKING:
    from .cscg import CSCGWorldModel


@dataclass
class ObjectInstance:
    """An object detected in a specific place."""
    class_name: str
    class_id: int
    position: Tuple[float, float]  # Normalized (cx, cy) in frame
    area: float  # Normalized area
    confidence: float
    visit_count: int = 1

    def update(self, pos: Tuple[float, float], area: float, conf: float):
        """Update object with new observation."""
        # Running average
        alpha = 1 / (self.visit_count + 1)
        self.position = (
            self.position[0] * (1 - alpha) + pos[0] * alpha,
            self.position[1] * (1 - alpha) + pos[1] * alpha,
        )
        self.area = self.area * (1 - alpha) + area * alpha
        self.confidence = max(self.confidence, conf)
        self.visit_count += 1


@dataclass
class SemanticPlace:
    """Semantic information about a CSCG place."""
    place_id: int  # Matches CSCG place ID

    # Room type classification
    room_type: Optional[str] = None
    room_type_confidence: float = 0.0
    room_type_scores: Dict[str, float] = field(default_factory=dict)

    # Objects observed at this place
    objects: Dict[str, ObjectInstance] = field(default_factory=dict)

    # Visit statistics
    visit_count: int = 0


@dataclass
class SemanticPrior:
    """Prior beliefs about what the environment should contain."""

    # Expected room types
    room_types: List[str] = field(default_factory=lambda: [
        "living_room", "kitchen", "bedroom", "bathroom", "hallway", "garage"
    ])

    # Expected objects per room type
    objects_per_room: Dict[str, List[str]] = field(default_factory=lambda: {
        "living_room": ["couch", "tv", "chair", "table"],
        "kitchen": ["refrigerator", "oven", "sink", "dining table", "chair"],
        "bedroom": ["bed", "chair", "tv"],
        "bathroom": ["toilet", "sink"],
        "hallway": ["door"],
        "garage": ["car", "bicycle"],
    })

    # Reference images for room type classification (ORB-based)
    # room_type -> list of reference image paths
    reference_images: Dict[str, List[str]] = field(default_factory=dict)

    # ORB descriptors for reference images (computed on load)
    reference_descriptors: Dict[str, List[np.ndarray]] = field(default_factory=dict)


class SemanticWorldModel:
    """
    Semantic layer that wraps CSCG world model.

    CSCG handles:
    - Place recognition (ORB keyframes)
    - Clone state disambiguation
    - Transition learning
    - EFE epistemic term

    This layer adds:
    - Room type classification per place
    - Object tracking per place
    - EFE pragmatic term (room/object priors)
    """

    # YOLO class names (COCO)
    YOLO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]

    def __init__(
        self,
        cscg: 'CSCGWorldModel',
        prior: Optional[SemanticPrior] = None,
        pragmatic_weight: float = 0.5,
    ):
        """
        Initialize semantic layer on top of CSCG.

        Args:
            cscg: CSCG world model (handles place recognition)
            prior: Semantic prior (expected rooms and objects)
            pragmatic_weight: Weight for pragmatic vs epistemic EFE (0-1)
        """
        self.cscg = cscg
        self.prior = prior or SemanticPrior()
        self.pragmatic_weight = pragmatic_weight
        self.n_actions = cscg.n_actions

        # ORB for room type classification (matches against reference images)
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.match_threshold = 0.75  # Lowe's ratio test threshold

        # Semantic info per CSCG place
        self.places: Dict[int, SemanticPlace] = {}

        # Frame counter
        self._frame_count = 0

        # Last localization result from CSCG
        self._last_cscg_result = None

        # Load reference images if provided
        if self.prior.reference_images:
            self._load_reference_images()

    def _load_reference_images(self):
        """Load and compute ORB descriptors for reference images."""
        print("Loading reference images for room type classification...")

        for room_type, paths in self.prior.reference_images.items():
            self.prior.reference_descriptors[room_type] = []

            for path in paths:
                img = cv2.imread(path)
                if img is None:
                    print(f"  Warning: Could not load {path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, desc = self.orb.detectAndCompute(gray, None)

                if desc is not None:
                    self.prior.reference_descriptors[room_type].append(desc)
                    print(f"  Loaded {path}: {len(kp)} features")

        print(f"Loaded references for {len(self.prior.reference_descriptors)} room types")

    def update(
        self,
        frame: np.ndarray,
        action_taken: int,
        observation_token: Optional[any] = None,
        yolo_detections: Optional[List] = None,
        room_hint: Optional[str] = None,
        debug: bool = False,
    ) -> Dict:
        """
        Update semantic layer with new observation.

        This calls CSCG for place recognition, then adds semantic info on top.

        Args:
            frame: BGR image
            action_taken: Movement action taken
            observation_token: Optional observation token for CSCG
            yolo_detections: Optional YOLO detection results
            room_hint: Optional ground truth room type (e.g., from simulator)
            debug: Print debug info

        Returns:
            Dict with localization result and EFE
        """
        self._frame_count += 1

        # 1. CSCG handles place recognition (bottom layer)
        cscg_result = self.cscg.localize(
            frame=frame,
            action_taken=action_taken,
            observation_token=observation_token,
            debug=debug,
        )
        self._last_cscg_result = cscg_result

        place_id = cscg_result.token
        place_conf = cscg_result.token_similarity
        is_new = cscg_result.new_token_discovered

        # 2. Ensure semantic place exists for this CSCG place
        if place_id not in self.places:
            self.places[place_id] = SemanticPlace(place_id=place_id)

        sem_place = self.places[place_id]
        sem_place.visit_count += 1

        # 3. Update object observations (semantic layer)
        if yolo_detections is not None:
            self._update_objects(place_id, yolo_detections, debug)
            # Infer room type from detected objects
            self._infer_room_from_objects(place_id, debug)

        # 4. Fallback room type classification
        if sem_place.room_type is None:
            if room_hint:
                # Use ground truth room hint (e.g., from simulator)
                self._update_room_from_hint(place_id, room_hint, debug)
            elif self.prior.reference_descriptors:
                # Use ORB matching against reference images
                self._update_room_type(place_id, frame, debug)

        # 5. Get place name from CSCG
        place_name = self.cscg.get_place_name(place_id)

        return {
            'place_id': place_id,
            'place_name': place_name,
            'room_type': sem_place.room_type,
            'confidence': place_conf,
            'is_new_place': is_new,
            'vfe': cscg_result.vfe,
            'n_places': self.cscg.n_locations,
            'objects_here': list(sem_place.objects.keys()),
            'cscg_result': cscg_result,
        }

    def _match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Match ORB descriptors and return confidence score."""
        if desc1 is None or desc2 is None:
            return 0.0

        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < self.match_threshold * m[1].distance:
                good_matches.append(m[0])

        # Confidence based on number of good matches
        confidence = len(good_matches) / max(len(desc1), 1)
        return min(confidence, 1.0)

    # Object -> Room type mapping for inference
    OBJECT_TO_ROOM = {
        # Bathroom indicators
        "toilet": "bathroom",
        "toothbrush": "bathroom",
        # Kitchen indicators
        "refrigerator": "kitchen",
        "oven": "kitchen",
        "microwave": "kitchen",
        "toaster": "kitchen",
        "sink": "kitchen",  # Could also be bathroom, but kitchen more common
        # Bedroom indicators
        "bed": "bedroom",
        # Living room indicators
        "couch": "living_room",
        "tv": "living_room",
        # Garage indicators
        "car": "garage",
        "bicycle": "garage",
        "motorcycle": "garage",
    }

    def _infer_room_from_objects(self, place_id: int, debug: bool):
        """Infer room type from detected objects at this place."""
        if place_id not in self.places:
            return

        place = self.places[place_id]
        if not place.objects:
            return

        # Count votes for each room type based on objects
        room_votes = {}
        for obj_name in place.objects.keys():
            room_type = self.OBJECT_TO_ROOM.get(obj_name)
            if room_type:
                if room_type not in room_votes:
                    room_votes[room_type] = 0
                # Weight by object confidence
                room_votes[room_type] += place.objects[obj_name].confidence

        if not room_votes:
            return

        # Find best room type
        best_type = max(room_votes, key=room_votes.get)
        total_weight = sum(room_votes.values())
        confidence = room_votes[best_type] / total_weight

        # Update room type if confident enough
        if confidence > 0.4:  # Lower threshold since object detection is reliable
            if place.room_type != best_type:
                place.room_type = best_type
                place.room_type_confidence = confidence
                if debug:
                    place_name = self.cscg.get_place_name(place_id)
                    objs = list(place.objects.keys())
                    print(f"  [ROOM] {place_name} -> {best_type} (from objects: {objs})")

    def _update_room_type(self, place_id: int, frame: np.ndarray, debug: bool):
        """Update room type classification for a place."""
        if place_id not in self.places:
            return

        place = self.places[place_id]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)

        if desc is None:
            return

        # Match against reference images for each room type
        scores = {}
        for room_type, ref_descs in self.prior.reference_descriptors.items():
            if not ref_descs:
                continue

            # Average match score across reference images
            match_scores = [self._match_descriptors(desc, rd) for rd in ref_descs]
            scores[room_type] = max(match_scores) if match_scores else 0.0

        if not scores:
            return

        # Update running average of scores
        alpha = 1 / (place.visit_count + 1)
        for room_type, score in scores.items():
            old_score = place.room_type_scores.get(room_type, 0.0)
            place.room_type_scores[room_type] = old_score * (1 - alpha) + score * alpha

        # Assign room type based on best score
        best_type = max(place.room_type_scores, key=place.room_type_scores.get)
        best_score = place.room_type_scores[best_type]

        if best_score > 0.3:  # Threshold for assignment
            place.room_type = best_type
            place.room_type_confidence = best_score

            if debug:
                place_name = self.cscg.get_place_name(place_id)
                print(f"  [ROOM] {place_name} classified as {best_type} (conf={best_score:.2f})")

    def _update_objects(self, place_id: int, detections: List, debug: bool):
        """Update object observations for a place."""
        place = self.places[place_id]

        for det in detections:
            # Extract detection info (assuming YOLO format)
            if hasattr(det, 'boxes') and len(det.boxes) > 0:
                for i, box in enumerate(det.boxes):
                    cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.5

                    if cls_id >= len(self.YOLO_CLASSES):
                        continue

                    class_name = self.YOLO_CLASSES[cls_id]

                    # Get position (normalized center)
                    if hasattr(box, 'xywhn'):
                        cx, cy = float(box.xywhn[0][0]), float(box.xywhn[0][1])
                        area = float(box.xywhn[0][2] * box.xywhn[0][3])
                    else:
                        cx, cy, area = 0.5, 0.5, 0.1

                    # Update or create object instance
                    if class_name in place.objects:
                        place.objects[class_name].update((cx, cy), area, conf)
                    else:
                        place.objects[class_name] = ObjectInstance(
                            class_name=class_name,
                            class_id=cls_id,
                            position=(cx, cy),
                            area=area,
                            confidence=conf,
                        )

                        if debug:
                            place_name = self.cscg.get_place_name(place_id)
                            print(f"  [OBJECT] Found {class_name} at {place_name}")

    def compute_efe(self, debug: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Compute Expected Free Energy for each action.

        EFE = (1 - w) * G_epistemic + w * G_pragmatic

        - G_epistemic: From CSCG (transition uncertainty)
        - G_pragmatic: From semantic priors (room/object coverage)

        Returns:
            (efe_per_action, info_dict)
        """
        info = {
            'epistemic': np.zeros(self.n_actions),
            'pragmatic': np.zeros(self.n_actions),
            'room_coverage': self._compute_room_coverage(),
            'object_coverage': self._compute_object_coverage(),
        }

        # 1. Get epistemic term from CSCG (transition uncertainty)
        # CSCG returns (best_action, action_probs) - convert to negative log (lower = better)
        _, cscg_probs = self.cscg.get_exploration_target()

        # Convert probabilities to EFE values (negative log-prob, so lower prob = higher EFE)
        # But we want to encourage exploration, so high CSCG prob = low epistemic EFE
        epsilon = 1e-10
        info['epistemic'] = -np.log(cscg_probs + epsilon)

        # 2. Compute pragmatic term (room/object prior satisfaction)
        current_place_id = self._last_cscg_result.token if self._last_cscg_result else None

        for action in range(self.n_actions):
            pragmatic = 0.0

            # Get transition predictions from CSCG
            if hasattr(self.cscg, 'predict_next_place'):
                next_places = self.cscg.predict_next_place(action)
            else:
                # Fallback: check semantic places we've seen via this action
                next_places = {}

            # Evaluate each possible destination
            for dest_id, prob in next_places.items() if next_places else []:
                sem_place = self.places.get(dest_id)
                if not sem_place:
                    # New place - high pragmatic value
                    pragmatic -= 1.0 * prob
                    continue

                # Reward for going to places with unknown room types
                if sem_place.room_type is None:
                    pragmatic -= 0.5 * prob

                # Reward for places with missing expected objects
                if sem_place.room_type:
                    expected_objs = set(self.prior.objects_per_room.get(
                        sem_place.room_type, []
                    ))
                    found_objs = set(sem_place.objects.keys())
                    missing = expected_objs - found_objs
                    pragmatic -= 0.2 * len(missing) * prob

            # Global pragmatic: prefer actions that lead to new rooms
            room_coverage = info['room_coverage']
            missing_rooms = set(self.prior.room_types) - set(room_coverage.keys())
            if missing_rooms:
                # General exploration bonus for movement actions
                if action in (1, 2, 3, 4):  # forward, back, left, right
                    pragmatic -= 0.3

            info['pragmatic'][action] = pragmatic

        # 3. Combine epistemic and pragmatic
        w = self.pragmatic_weight
        efe = (1 - w) * info['epistemic'] + w * info['pragmatic']

        if debug:
            print(f"  [EFE] epistemic={info['epistemic']}")
            print(f"  [EFE] pragmatic={info['pragmatic']}")
            print(f"  [EFE] total={efe} (w={w})")

        return efe, info

    def get_exploration_action(self, debug: bool = False) -> Tuple[int, str]:
        """
        Get best action for exploration (minimizes EFE).

        Returns:
            (action_idx, reason)
        """
        efe, info = self.compute_efe(debug)

        # Select action with lowest EFE
        best_action = int(np.argmin(efe))

        # Generate reason
        room_coverage = info['room_coverage']
        missing_rooms = set(self.prior.room_types) - set(room_coverage.keys())

        if missing_rooms:
            reason = f"Looking for: {', '.join(missing_rooms)}"
        elif info['epistemic'][best_action] < -0.5:
            reason = "Exploring unknown transition"
        else:
            reason = "Refining map"

        return best_action, reason

    def _compute_room_coverage(self) -> Dict[str, int]:
        """Compute which room types have been found."""
        coverage = {}
        for place in self.places.values():
            if place.room_type:
                if place.room_type not in coverage:
                    coverage[place.room_type] = 0
                coverage[place.room_type] += 1
        return coverage

    def _compute_object_coverage(self) -> Dict[str, Dict[str, bool]]:
        """Compute which expected objects have been found per room type."""
        coverage = {}

        for place in self.places.values():
            if not place.room_type:
                continue

            if place.room_type not in coverage:
                expected = self.prior.objects_per_room.get(place.room_type, [])
                coverage[place.room_type] = {obj: False for obj in expected}

            for obj_name in place.objects.keys():
                if obj_name in coverage[place.room_type]:
                    coverage[place.room_type][obj_name] = True

        return coverage

    def get_exploration_urgency(self) -> Tuple[float, str]:
        """
        Get exploration urgency based on semantic coverage.

        Returns:
            (urgency 0-1, reason)
        """
        # Room coverage
        room_coverage = self._compute_room_coverage()
        found_rooms = len(room_coverage)
        expected_rooms = len(self.prior.room_types)
        room_urgency = 1.0 - found_rooms / max(expected_rooms, 1)

        # Object coverage
        obj_coverage = self._compute_object_coverage()
        total_expected = 0
        total_found = 0
        for room_type, objs in obj_coverage.items():
            total_expected += len(objs)
            total_found += sum(objs.values())

        obj_urgency = 1.0 - total_found / max(total_expected, 1) if total_expected > 0 else 1.0

        # Combine
        urgency = 0.6 * room_urgency + 0.4 * obj_urgency

        # Reason
        missing_rooms = set(self.prior.room_types) - set(room_coverage.keys())
        if missing_rooms:
            reason = f"Missing rooms: {', '.join(missing_rooms)}"
        elif obj_urgency > 0.3:
            reason = "Looking for missing objects"
        else:
            reason = "Environment well-mapped"

        return urgency, reason

    def get_map_summary(self) -> str:
        """Get human-readable map summary."""
        lines = ["=== Semantic Map ==="]
        lines.append(f"CSCG places: {self.cscg.n_locations}, Semantic places: {len(self.places)}")

        for pid, sem_place in sorted(self.places.items()):
            place_name = self.cscg.get_place_name(pid)
            room_str = sem_place.room_type or "unknown"
            obj_str = ", ".join(sem_place.objects.keys()) or "none"
            lines.append(f"  {place_name} ({room_str}): [{obj_str}]")

        # Coverage summary
        room_cov = self._compute_room_coverage()
        lines.append(f"\nRooms found: {list(room_cov.keys())}")
        lines.append(f"Missing: {set(self.prior.room_types) - set(room_cov.keys())}")

        # Object coverage
        obj_cov = self._compute_object_coverage()
        for room_type, objs in obj_cov.items():
            found = [o for o, f in objs.items() if f]
            missing = [o for o, f in objs.items() if not f]
            if found or missing:
                lines.append(f"  {room_type}: found={found}, missing={missing}")

        return "\n".join(lines)

    def reset(self):
        """Reset the semantic layer (and optionally CSCG)."""
        self.places.clear()
        self._frame_count = 0
        self._last_cscg_result = None
        # Note: CSCG reset should be called separately if needed
        # self.cscg.reset()

    @property
    def current_place_id(self) -> Optional[int]:
        """Get current place ID from last CSCG result."""
        if self._last_cscg_result:
            return self._last_cscg_result.token
        return None

    @property
    def n_places(self) -> int:
        """Get number of CSCG places."""
        return self.cscg.n_locations
