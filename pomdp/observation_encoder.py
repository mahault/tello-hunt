"""
Observation encoder for converting YOLO detections to fixed-size tokens.

Converts variable-length YOLO detection lists into fixed-size observation
vectors suitable for JIT-compiled POMDP inference.
"""

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .config import (
    COCO_TO_TYPE, N_OBJECT_TYPES, TYPE_NAMES,
    N_OBS_LEVELS, CONF_THRESHOLD_LOW, CONF_THRESHOLD_HIGH,
    PERSON_LEFT_THRESHOLD, PERSON_RIGHT_THRESHOLD, PERSON_CLOSE_THRESHOLD
)


@dataclass
class ObservationToken:
    """
    Fixed-size observation token from YOLO detections.

    All fields have fixed sizes for JIT compatibility.
    """
    # Object detection histogram (N_OBJECT_TYPES,)
    # Each element: 0 = absent, 1 = low_conf, 2 = high_conf
    object_levels: np.ndarray

    # Object counts (N_OBJECT_TYPES,) - clipped to max 5
    object_counts: np.ndarray

    # Max confidence per object type (N_OBJECT_TYPES,)
    object_max_conf: np.ndarray

    # Average position per object type (N_OBJECT_TYPES, 2) - normalized [-1, 1]
    object_avg_pos: np.ndarray

    # Average area per object type (N_OBJECT_TYPES,) - normalized [0, 1]
    object_avg_area: np.ndarray

    # Person-specific features (for human search POMDP)
    person_detected: bool
    person_area: float      # 0-1, normalized
    person_cx: float        # -1 to 1, horizontal position
    person_cy: float        # -1 to 1, vertical position
    person_conf: float      # 0-1, confidence

    # Discretized person observation index
    person_obs_idx: int     # 0=not_detected, 1=left, 2=center, 3=right, 4=close

    def to_signature_vector(self) -> np.ndarray:
        """
        Create a compact signature vector for location matching.

        Returns fixed-size vector representing the observation.
        """
        # Combine object presence and confidence into signature
        # Weight by confidence for better matching
        signature = self.object_levels.astype(np.float32) * self.object_max_conf
        return signature

    def to_jax(self) -> Dict[str, jnp.ndarray]:
        """Convert to JAX arrays for JIT functions."""
        return {
            'object_levels': jnp.array(self.object_levels),
            'object_counts': jnp.array(self.object_counts),
            'object_max_conf': jnp.array(self.object_max_conf),
            'object_avg_pos': jnp.array(self.object_avg_pos),
            'object_avg_area': jnp.array(self.object_avg_area),
            'person_detected': self.person_detected,
            'person_area': self.person_area,
            'person_cx': self.person_cx,
            'person_obs_idx': self.person_obs_idx,
        }


def encode_yolo_detections(
    boxes: Any,  # YOLO Boxes object
    model_names: Dict[int, str],
    frame_width: int,
    frame_height: int,
    conf_threshold: float = 0.45
) -> ObservationToken:
    """
    Encode YOLO detections into fixed-size observation token.

    This runs on CPU (not JIT) since it processes variable-length YOLO output.

    Args:
        boxes: YOLO detection boxes (res.boxes from model inference)
        model_names: model.names dict mapping class IDs to names
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        conf_threshold: Minimum confidence to include detection

    Returns:
        ObservationToken with fixed-size arrays
    """
    frame_area = float(frame_width * frame_height)

    # Initialize accumulators
    object_counts = np.zeros(N_OBJECT_TYPES, dtype=np.float32)
    object_conf_sum = np.zeros(N_OBJECT_TYPES, dtype=np.float32)
    object_x_sum = np.zeros(N_OBJECT_TYPES, dtype=np.float32)
    object_y_sum = np.zeros(N_OBJECT_TYPES, dtype=np.float32)
    object_area_sum = np.zeros(N_OBJECT_TYPES, dtype=np.float32)
    object_max_conf = np.zeros(N_OBJECT_TYPES, dtype=np.float32)

    # Person-specific tracking (keep best detection)
    person_detected = False
    person_area = 0.0
    person_cx = 0.0
    person_cy = 0.0
    person_conf = 0.0

    # Process each detection
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < conf_threshold:
            continue

        # Check if this is a tracked object type
        if cls_id not in COCO_TO_TYPE:
            continue

        type_idx = COCO_TO_TYPE[cls_id]

        # Extract bounding box
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1
        area = (bw * bh) / frame_area

        # Normalize coordinates to [-1, 1]
        cx_norm = (cx / frame_width) * 2 - 1
        cy_norm = (cy / frame_height) * 2 - 1

        # Accumulate statistics
        object_counts[type_idx] += 1
        object_conf_sum[type_idx] += conf
        object_x_sum[type_idx] += cx_norm
        object_y_sum[type_idx] += cy_norm
        object_area_sum[type_idx] += area
        object_max_conf[type_idx] = max(object_max_conf[type_idx], conf)

        # Special handling for person (type_idx 0)
        if type_idx == 0 and conf > person_conf:
            person_detected = True
            person_area = area
            person_cx = cx_norm
            person_cy = cy_norm
            person_conf = conf

    # Compute averages
    counts_safe = np.maximum(object_counts, 1.0)
    object_avg_x = object_x_sum / counts_safe
    object_avg_y = object_y_sum / counts_safe
    object_avg_area = object_area_sum / counts_safe

    # Stack positions
    object_avg_pos = np.stack([object_avg_x, object_avg_y], axis=1)

    # Clip counts to max 5
    object_counts = np.minimum(object_counts, 5.0)

    # Discretize to observation levels
    object_levels = np.zeros(N_OBJECT_TYPES, dtype=np.int32)
    for i in range(N_OBJECT_TYPES):
        if object_counts[i] > 0:
            if object_max_conf[i] >= CONF_THRESHOLD_HIGH:
                object_levels[i] = 2  # high_conf
            else:
                object_levels[i] = 1  # low_conf
        # else: 0 = absent

    # Discretize person observation
    person_obs_idx = discretize_person_obs(
        person_detected, person_cx, person_area
    )

    return ObservationToken(
        object_levels=object_levels,
        object_counts=object_counts,
        object_max_conf=object_max_conf,
        object_avg_pos=object_avg_pos,
        object_avg_area=object_avg_area,
        person_detected=person_detected,
        person_area=person_area,
        person_cx=person_cx,
        person_cy=person_cy,
        person_conf=person_conf,
        person_obs_idx=person_obs_idx,
    )


def discretize_person_obs(
    detected: bool,
    cx: float,
    area: float
) -> int:
    """
    Discretize person observation into categorical index.

    Returns:
        0: not_detected
        1: detected_left
        2: detected_center
        3: detected_right
        4: detected_close
    """
    if not detected:
        return 0

    # Close takes priority
    if area > PERSON_CLOSE_THRESHOLD:
        return 4

    # Then check horizontal position
    if cx < PERSON_LEFT_THRESHOLD:
        return 1  # left
    elif cx > PERSON_RIGHT_THRESHOLD:
        return 3  # right
    else:
        return 2  # center


def create_empty_observation() -> ObservationToken:
    """Create an empty observation token (no detections)."""
    return ObservationToken(
        object_levels=np.zeros(N_OBJECT_TYPES, dtype=np.int32),
        object_counts=np.zeros(N_OBJECT_TYPES, dtype=np.float32),
        object_max_conf=np.zeros(N_OBJECT_TYPES, dtype=np.float32),
        object_avg_pos=np.zeros((N_OBJECT_TYPES, 2), dtype=np.float32),
        object_avg_area=np.zeros(N_OBJECT_TYPES, dtype=np.float32),
        person_detected=False,
        person_area=0.0,
        person_cx=0.0,
        person_cy=0.0,
        person_conf=0.0,
        person_obs_idx=0,
    )


def observation_to_text(obs: ObservationToken) -> str:
    """
    Convert observation to human-readable text.

    Useful for debugging and visualization.
    """
    lines = []

    # List detected objects
    detected = []
    for i, level in enumerate(obs.object_levels):
        if level > 0:
            conf_str = "high" if level == 2 else "low"
            count = int(obs.object_counts[i])
            detected.append(f"{TYPE_NAMES[i]}({count}, {conf_str})")

    if detected:
        lines.append(f"Objects: {', '.join(detected)}")
    else:
        lines.append("Objects: none")

    # Person info
    if obs.person_detected:
        pos_names = ['not_detected', 'left', 'center', 'right', 'close']
        lines.append(
            f"Person: area={obs.person_area:.2f}, "
            f"pos={pos_names[obs.person_obs_idx]}, "
            f"conf={obs.person_conf:.2f}"
        )
    else:
        lines.append("Person: not detected")

    return "\n".join(lines)
