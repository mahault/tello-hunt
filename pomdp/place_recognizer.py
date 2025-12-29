"""
ORB-based Place Recognition for Navigation.

Uses local feature matching (ORB keypoints + descriptors) instead of
semantic embeddings (CLIP/DINOv2) for place recognition.

Why this works better for navigation:
- Local features are viewpoint-robust
- Captures corners, edges, texture (not semantics)
- Naturally supports visual odometry + place recognition
- "Have I been here before?" vs "What room is this?"
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class Keyframe:
    """A stored keyframe for place recognition."""
    id: int
    name: str  # Human-readable name (e.g., "Living Room")
    descriptors: np.ndarray  # ORB descriptors (N, 32)
    keypoints_xy: np.ndarray  # Keypoint positions (N, 2)
    timestamp: int = 0
    visit_count: int = 1
    thumbnail: Optional[np.ndarray] = None


class ORBPlaceRecognizer:
    """
    Place recognition using ORB features and keyframe matching.

    Supports:
    - Online learning (discover places during exploration)
    - Preloading (bootstrap with known room images)
    """

    def __init__(
        self,
        n_features: int = 500,
        match_threshold: float = 0.75,  # Lowe's ratio test
        min_matches: int = 12,  # Minimum matches to recognize
        keyframe_cooldown: int = 15,  # Frames between new keyframes
        max_keyframes: int = 50,
    ):
        self.n_features = n_features
        self.match_threshold = match_threshold
        self.min_matches = min_matches
        self.keyframe_cooldown = keyframe_cooldown
        self.max_keyframes = max_keyframes

        # ORB detector
        self.orb = cv2.ORB_create(nfeatures=n_features)

        # Brute force matcher with Hamming distance
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Keyframe database
        self.keyframes: List[Keyframe] = []
        self._next_id: int = 0

        # State
        self._frames_since_keyframe: int = 0
        self._current_place: int = -1
        self._current_name: str = "Unknown"
        self._frame_count: int = 0

    def preload_room(
        self,
        frames: List[np.ndarray],
        room_name: str,
        debug: bool = False,
    ) -> int:
        """
        Preload keyframes for a known room.

        Args:
            frames: List of BGR images from the room
            room_name: Human-readable name
            debug: Print debug info

        Returns:
            place_id assigned to this room
        """
        all_descriptors = []
        all_keypoints = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            kp, desc = self.orb.detectAndCompute(gray, None)

            if desc is not None and len(kp) > 10:
                all_descriptors.append(desc)
                all_keypoints.extend([(p.pt[0], p.pt[1]) for p in kp])

        if not all_descriptors:
            if debug:
                print(f"  [ORB] Warning: No features found for {room_name}")
            return -1

        # Combine descriptors (take best from each view)
        combined_desc = np.vstack(all_descriptors)

        # Limit to n_features (keep diverse set)
        if len(combined_desc) > self.n_features:
            indices = np.random.choice(len(combined_desc), self.n_features, replace=False)
            combined_desc = combined_desc[indices]

        keypoints_xy = np.array(all_keypoints[:len(combined_desc)])

        # Create thumbnail from first frame
        thumbnail = cv2.resize(frames[0], (80, 60))
        if len(thumbnail.shape) == 3:
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)

        # Create keyframe
        kf = Keyframe(
            id=self._next_id,
            name=room_name,
            descriptors=combined_desc,
            keypoints_xy=keypoints_xy,
            thumbnail=thumbnail,
        )
        self.keyframes.append(kf)

        place_id = self._next_id
        self._next_id += 1

        if debug:
            print(f"  [ORB] Preloaded '{room_name}' as place {place_id} ({len(combined_desc)} features)")

        return place_id

    def recognize(
        self,
        frame: np.ndarray,
        allow_new: bool = True,
        debug: bool = False,
    ) -> Tuple[int, str, float, bool]:
        """
        Recognize place from frame.

        Args:
            frame: BGR image
            allow_new: Allow creating new keyframes
            debug: Print debug info

        Returns:
            (place_id, place_name, confidence, is_new_place)
        """
        self._frame_count += 1
        self._frames_since_keyframe += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            if debug:
                print(f"  [ORB] Low features, staying at {self._current_name}")
            return self._current_place, self._current_name, 0.0, False

        # First keyframe
        if len(self.keyframes) == 0:
            if allow_new:
                place_id = self._create_keyframe(descriptors, keypoints, gray, "Place_0")
                if debug:
                    print(f"  [ORB] First keyframe: {self._current_name}")
                return place_id, self._current_name, 1.0, True
            return -1, "Unknown", 0.0, False

        # Find best match
        best_id, best_name, best_matches, confidence = self._find_best_match(descriptors)

        is_new = False

        if best_matches >= self.min_matches:
            # Good match
            place_id = best_id
            place_name = best_name

            # Update visit count
            for kf in self.keyframes:
                if kf.id == place_id:
                    kf.visit_count += 1
                    break

            if debug and (self._frame_count % 30 == 0 or place_id != self._current_place):
                print(f"  [ORB] Matched '{place_name}' ({best_matches} matches, conf={confidence:.2f})")

        elif allow_new and self._frames_since_keyframe >= self.keyframe_cooldown:
            # Create new keyframe
            place_id = self._create_keyframe(descriptors, keypoints, gray, f"Place_{self._next_id}")
            place_name = self._current_name
            confidence = 1.0
            is_new = True

            if debug:
                print(f"  [ORB] New place '{place_name}' (best was {best_matches} matches)")

        else:
            # Stay at current
            place_id = self._current_place if self._current_place >= 0 else best_id
            place_name = self._current_name if self._current_place >= 0 else best_name

            if debug and self._frame_count % 60 == 0:
                print(f"  [ORB] Uncertain, staying at '{place_name}'")

        self._current_place = place_id
        self._current_name = place_name

        return place_id, place_name, confidence, is_new

    def _find_best_match(self, descriptors: np.ndarray) -> Tuple[int, str, int, float]:
        """Find best matching keyframe."""
        best_id = -1
        best_name = "Unknown"
        best_matches = 0
        best_confidence = 0.0

        for kf in self.keyframes:
            matches = self.bf.knnMatch(descriptors, kf.descriptors, k=2)

            # Lowe's ratio test
            good = []
            for m in matches:
                if len(m) == 2 and m[0].distance < self.match_threshold * m[1].distance:
                    good.append(m[0])

            n_good = len(good)
            confidence = min(1.0, n_good / (self.min_matches * 2))

            if n_good > best_matches:
                best_id = kf.id
                best_name = kf.name
                best_matches = n_good
                best_confidence = confidence

        return best_id, best_name, best_matches, best_confidence

    def _create_keyframe(
        self,
        descriptors: np.ndarray,
        keypoints: List,
        gray: np.ndarray,
        name: str,
    ) -> int:
        """Create new keyframe."""
        keypoints_xy = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        thumbnail = cv2.resize(gray, (80, 60))

        kf = Keyframe(
            id=self._next_id,
            name=name,
            descriptors=descriptors.copy(),
            keypoints_xy=keypoints_xy,
            thumbnail=thumbnail,
        )

        if len(self.keyframes) >= self.max_keyframes:
            # Remove least visited (not current)
            min_idx = min(
                range(len(self.keyframes)),
                key=lambda i: (self.keyframes[i].id == self._current_place, self.keyframes[i].visit_count)
            )
            self.keyframes.pop(min_idx)

        self.keyframes.append(kf)

        place_id = self._next_id
        self._next_id += 1
        self._frames_since_keyframe = 0
        self._current_place = place_id
        self._current_name = name

        return place_id

    @property
    def n_places(self) -> int:
        return len(self.keyframes)

    def get_place_name(self, place_id: int) -> str:
        """Get name for place ID."""
        for kf in self.keyframes:
            if kf.id == place_id:
                return kf.name
        return "Unknown"

    def reset(self):
        """Reset (keeps preloaded keyframes)."""
        self._frames_since_keyframe = 0
        self._current_place = 0 if self.keyframes else -1
        self._current_name = self.keyframes[0].name if self.keyframes else "Unknown"
        self._frame_count = 0

    def clear(self):
        """Clear everything including preloaded keyframes."""
        self.keyframes.clear()
        self._next_id = 0
        self.reset()

    def __repr__(self) -> str:
        places = [kf.name for kf in self.keyframes]
        return f"ORBPlaceRecognizer(places={places}, current='{self._current_name}')"
