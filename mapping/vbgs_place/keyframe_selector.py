"""
Keyframe selection for VBGS place models.

Decides when to add new keyframes based on viewpoint change
and embedding distance.
"""

import numpy as np
import time
from typing import Optional


class KeyframeSelector:
    """
    Selects keyframes for place model updates.

    A frame becomes a keyframe when:
    1. Sufficient time has passed since last keyframe
    2. Embedding is different enough from last keyframe
    3. Image content has changed significantly

    Attributes:
        min_interval: Minimum seconds between keyframes
        embedding_threshold: Minimum embedding distance for new keyframe
        last_keyframe_time: Timestamp of last keyframe
        last_keyframe_embedding: Embedding of last keyframe
    """

    def __init__(
        self,
        min_interval: float = 0.5,
        embedding_threshold: float = 0.15,
        max_keyframes_per_place: int = 100,
    ):
        """
        Initialize keyframe selector.

        Args:
            min_interval: Minimum seconds between keyframes
            embedding_threshold: Minimum 1 - cosine_similarity for new keyframe
            max_keyframes_per_place: Maximum keyframes per place
        """
        self.min_interval = min_interval
        self.embedding_threshold = embedding_threshold
        self.max_keyframes_per_place = max_keyframes_per_place

        # State
        self.last_keyframe_time = 0.0
        self.last_keyframe_embedding: Optional[np.ndarray] = None
        self.keyframe_count = 0

    def should_add_keyframe(
        self,
        embedding: np.ndarray,
        place_keyframe_count: int = 0,
    ) -> bool:
        """
        Check if current frame should be a keyframe.

        Args:
            embedding: Current frame's CLIP embedding
            place_keyframe_count: Current keyframes for this place

        Returns:
            True if should add keyframe
        """
        current_time = time.time()

        # Check time interval
        if current_time - self.last_keyframe_time < self.min_interval:
            return False

        # Check max keyframes
        if place_keyframe_count >= self.max_keyframes_per_place:
            return False

        # First keyframe always added
        if self.last_keyframe_embedding is None:
            return True

        # Check embedding distance
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        last = self.last_keyframe_embedding

        # Normalize
        emb_norm = np.linalg.norm(embedding)
        last_norm = np.linalg.norm(last)

        if emb_norm > 0 and last_norm > 0:
            similarity = np.dot(embedding, last) / (emb_norm * last_norm)
            distance = 1.0 - similarity

            if distance >= self.embedding_threshold:
                return True

        return False

    def mark_keyframe(self, embedding: np.ndarray):
        """
        Mark that a keyframe was added.

        Call this after successfully adding a keyframe.
        """
        self.last_keyframe_time = time.time()
        self.last_keyframe_embedding = np.asarray(embedding, dtype=np.float32).flatten()
        self.keyframe_count += 1

    def reset(self):
        """Reset selector state."""
        self.last_keyframe_time = 0.0
        self.last_keyframe_embedding = None
        self.keyframe_count = 0

    def __repr__(self) -> str:
        return (f"KeyframeSelector(interval={self.min_interval}s, "
                f"threshold={self.embedding_threshold}, "
                f"count={self.keyframe_count})")


class AdaptiveKeyframeSelector(KeyframeSelector):
    """
    Adaptive keyframe selector that adjusts threshold based on place novelty.

    Adds more keyframes to novel places and fewer to established ones.
    """

    def __init__(
        self,
        min_interval: float = 0.5,
        base_threshold: float = 0.15,
        novel_threshold: float = 0.05,  # Lower threshold for novel places
        established_threshold: float = 0.25,  # Higher threshold for established
        established_count: int = 20,  # Keyframes to be "established"
        max_keyframes_per_place: int = 100,
    ):
        """
        Initialize adaptive keyframe selector.

        Args:
            min_interval: Minimum seconds between keyframes
            base_threshold: Base embedding distance threshold
            novel_threshold: Threshold for novel places (more keyframes)
            established_threshold: Threshold for established places (fewer keyframes)
            established_count: Number of keyframes to be "established"
            max_keyframes_per_place: Maximum keyframes per place
        """
        super().__init__(min_interval, base_threshold, max_keyframes_per_place)
        self.novel_threshold = novel_threshold
        self.established_threshold = established_threshold
        self.established_count = established_count

    def should_add_keyframe(
        self,
        embedding: np.ndarray,
        place_keyframe_count: int = 0,
    ) -> bool:
        """
        Check if current frame should be a keyframe with adaptive threshold.
        """
        current_time = time.time()

        # Check time interval
        if current_time - self.last_keyframe_time < self.min_interval:
            return False

        # Check max keyframes
        if place_keyframe_count >= self.max_keyframes_per_place:
            return False

        # First keyframe always added
        if self.last_keyframe_embedding is None:
            return True

        # Adaptive threshold based on place maturity
        if place_keyframe_count < 5:
            threshold = self.novel_threshold
        elif place_keyframe_count >= self.established_count:
            threshold = self.established_threshold
        else:
            # Interpolate
            t = place_keyframe_count / self.established_count
            threshold = self.novel_threshold + t * (self.established_threshold - self.novel_threshold)

        # Check embedding distance
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        last = self.last_keyframe_embedding

        emb_norm = np.linalg.norm(embedding)
        last_norm = np.linalg.norm(last)

        if emb_norm > 0 and last_norm > 0:
            similarity = np.dot(embedding, last) / (emb_norm * last_norm)
            distance = 1.0 - similarity

            if distance >= threshold:
                return True

        return False
