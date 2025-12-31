"""
Collision Avoidance Module for Monocular Camera Navigation.

Implements a hybrid approach combining:
1. Optical Flow Time-to-Contact (TTC) - fast reflex for approaching surfaces
2. Monocular Depth Gate - confirmatory gate using depth percentiles
3. Yaw Scan Escape - find open direction when stuck

This works with a single RGB camera (no depth sensor required).
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Collision risk levels."""
    NONE = 0      # Safe to proceed
    LOW = 1       # Proceed with caution
    MEDIUM = 2    # Slow down, consider turning
    HIGH = 3      # Stop, must avoid
    CRITICAL = 4  # Emergency stop


@dataclass
class CollisionState:
    """Current collision avoidance state."""
    risk_level: RiskLevel
    flow_risk: float      # 0-1, from optical flow TTC
    depth_risk: float     # 0-1, from monocular depth
    combined_risk: float  # 0-1, hybrid
    ttc_estimate: float   # Time to contact in seconds (inf = safe)
    escape_direction: Optional[int] = None  # Best yaw direction if blocked
    debug_info: Dict = None


class OpticalFlowTTC:
    """
    Time-to-Contact estimation using optical flow.

    Detects approaching surfaces by measuring flow expansion/divergence.
    High expansion = low TTC = collision imminent.
    """

    def __init__(
        self,
        flow_method: str = "farneback",
        roi_fraction: float = 0.4,  # Center ROI for forward collision
        ttc_threshold: float = 1.5,  # Seconds - below this is dangerous
    ):
        self.flow_method = flow_method
        self.roi_fraction = roi_fraction
        self.ttc_threshold = ttc_threshold

        self._prev_gray: Optional[np.ndarray] = None
        self._flow_ema: float = 0.0
        self._ema_alpha: float = 0.3

        # Farneback parameters
        self._farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=0
        )

    def compute_ttc(self, frame: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Compute time-to-contact from optical flow.

        Args:
            frame: BGR image

        Returns:
            (ttc_seconds, risk_0_to_1, flow_visualization)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # First frame - no flow yet
        if self._prev_gray is None:
            self._prev_gray = gray
            return float('inf'), 0.0, np.zeros((h, w), dtype=np.uint8)

        # Compute optical flow
        if self.flow_method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None, **self._farneback_params
            )
        else:
            # Lucas-Kanade sparse flow (faster but less robust)
            flow = self._compute_sparse_flow(self._prev_gray, gray)

        self._prev_gray = gray.copy()

        if flow is None:
            return float('inf'), 0.0, np.zeros((h, w), dtype=np.uint8)

        # Extract center ROI (where we're heading)
        roi_h = int(h * self.roi_fraction)
        roi_w = int(w * self.roi_fraction)
        y1, y2 = (h - roi_h) // 2, (h + roi_h) // 2
        x1, x2 = (w - roi_w) // 2, (w + roi_w) // 2

        flow_roi = flow[y1:y2, x1:x2]

        # Compute flow magnitude and expansion
        fx, fy = flow_roi[:, :, 0], flow_roi[:, :, 1]
        magnitude = np.sqrt(fx**2 + fy**2)

        # Compute divergence (expansion) - key for TTC
        # Positive divergence = expansion = approaching surface
        div_x = np.gradient(fx, axis=1)
        div_y = np.gradient(fy, axis=0)
        divergence = div_x + div_y

        # Focus on positive divergence (expansion)
        expansion = np.clip(divergence, 0, None)

        # Use 80th percentile of expansion as robust metric
        if expansion.size > 0:
            expansion_metric = np.percentile(expansion, 80)
        else:
            expansion_metric = 0.0

        # Apply EMA smoothing
        self._flow_ema = (
            self._ema_alpha * expansion_metric +
            (1 - self._ema_alpha) * self._flow_ema
        )

        # Estimate TTC (simplified - assumes constant approach speed)
        # TTC ~ 1 / expansion_rate
        if self._flow_ema > 0.1:
            ttc = 10.0 / self._flow_ema  # Scale factor for reasonable TTC
            ttc = np.clip(ttc, 0.1, 100.0)
        else:
            ttc = float('inf')

        # Convert to risk (0 = safe, 1 = danger)
        if ttc < self.ttc_threshold:
            risk = np.clip(1.0 - (ttc / self.ttc_threshold), 0, 1)
        else:
            risk = 0.0

        # Create flow visualization
        flow_vis = self._visualize_flow(flow, h, w)

        return ttc, risk, flow_vis

    def _compute_sparse_flow(self, prev: np.ndarray, curr: np.ndarray) -> Optional[np.ndarray]:
        """Compute sparse optical flow using Lucas-Kanade."""
        # Detect features in previous frame
        features = cv2.goodFeaturesToTrack(
            prev, maxCorners=100, qualityLevel=0.3, minDistance=7
        )

        if features is None:
            return None

        # Track features
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, curr, features, None
        )

        if next_pts is None:
            return None

        # Create dense-ish flow from sparse points
        h, w = prev.shape
        flow = np.zeros((h, w, 2), dtype=np.float32)

        for i, (old, new) in enumerate(zip(features, next_pts)):
            if status[i]:
                ox, oy = old.ravel()
                nx, ny = new.ravel()
                # Simple nearest-neighbor interpolation
                x, y = int(ox), int(oy)
                if 0 <= x < w and 0 <= y < h:
                    flow[y, x] = [nx - ox, ny - oy]

        return flow

    def _visualize_flow(self, flow: np.ndarray, h: int, w: int) -> np.ndarray:
        """Create HSV visualization of optical flow."""
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        magnitude = np.sqrt(fx**2 + fy**2)
        angle = np.arctan2(fy, fx)

        # HSV: Hue = direction, Saturation = 1, Value = magnitude
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = np.clip(magnitude * 10, 0, 255).astype(np.uint8)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def reset(self):
        """Reset flow state."""
        self._prev_gray = None
        self._flow_ema = 0.0


class CollisionAvoidance:
    """
    Hybrid collision avoidance combining optical flow and monocular depth.

    Provides a state machine for safe navigation:
    - Assess risk from multiple sources
    - Trigger escape maneuvers when blocked
    - Yaw scan to find open directions
    """

    def __init__(
        self,
        depth_estimator=None,
        use_flow: bool = True,
        use_depth: bool = True,
        flow_weight: float = 0.6,  # Weight for flow vs depth
        risk_thresholds: Dict[str, float] = None,
    ):
        """
        Initialize collision avoidance.

        Args:
            depth_estimator: Optional DepthEstimator instance
            use_flow: Enable optical flow TTC
            use_depth: Enable monocular depth gate
            flow_weight: Weight for flow in combined risk (0-1)
            risk_thresholds: Custom thresholds for risk levels
        """
        self.depth_estimator = depth_estimator
        self.use_flow = use_flow
        self.use_depth = use_depth
        self.flow_weight = flow_weight

        # Risk thresholds
        self.thresholds = risk_thresholds or {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9,
        }

        # Initialize optical flow
        if use_flow:
            self.flow_ttc = OpticalFlowTTC()
        else:
            self.flow_ttc = None

        # State tracking
        self._consecutive_high_risk: int = 0
        self._escape_in_progress: bool = False
        self._yaw_scan_data: List[Tuple[float, float]] = []  # (yaw, risk) pairs
        self._current_scan_yaw: float = 0.0

        # Backoff rotation state (rotate 180° before backing off)
        self._backoff_rotation_frames: int = 0
        self._backoff_rotation_target: int = 18  # ~180° at 10°/frame
        self._backoff_turn_direction: int = 4  # Default: turn right

        # Control limits (for safety)
        self.max_speed = 30  # RC control units
        self.max_yaw_rate = 30
        self.burst_duration = 0.3  # seconds per movement burst

    def assess_risk(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray = None,
        debug: bool = False,
    ) -> CollisionState:
        """
        Assess collision risk from current frame.

        Args:
            frame: BGR image
            depth_map: Optional pre-computed depth map
            debug: Print debug info

        Returns:
            CollisionState with risk assessment
        """
        flow_risk = 0.0
        depth_risk = 0.0
        ttc = float('inf')
        flow_vis = None

        # 1. Optical flow TTC (fast reflex)
        if self.use_flow and self.flow_ttc is not None:
            ttc, flow_risk, flow_vis = self.flow_ttc.compute_ttc(frame)

        # 2. Monocular depth gate (confirmatory)
        if self.use_depth and self.depth_estimator is not None:
            if depth_map is None:
                depth_map = self.depth_estimator.estimate(frame)

            is_clear, distance = self.depth_estimator.is_path_clear(
                depth_map, direction="forward", threshold=0.3
            )
            depth_risk = 1.0 - distance  # Invert: low distance = high risk

        # 3. Combine risks (hybrid)
        if self.use_flow and self.use_depth:
            combined_risk = (
                self.flow_weight * flow_risk +
                (1 - self.flow_weight) * depth_risk
            )
        elif self.use_flow:
            combined_risk = flow_risk
        elif self.use_depth:
            combined_risk = depth_risk
        else:
            combined_risk = 0.0

        # 4. Determine risk level
        if combined_risk >= self.thresholds['critical']:
            risk_level = RiskLevel.CRITICAL
        elif combined_risk >= self.thresholds['high']:
            risk_level = RiskLevel.HIGH
        elif combined_risk >= self.thresholds['medium']:
            risk_level = RiskLevel.MEDIUM
        elif combined_risk >= self.thresholds['low']:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.NONE

        # 5. Track consecutive high risk
        if risk_level.value >= RiskLevel.HIGH.value:
            self._consecutive_high_risk += 1
        else:
            self._consecutive_high_risk = 0

        # Create state
        state = CollisionState(
            risk_level=risk_level,
            flow_risk=flow_risk,
            depth_risk=depth_risk,
            combined_risk=combined_risk,
            ttc_estimate=ttc,
            debug_info={
                'flow_vis': flow_vis,
                'consecutive_high': self._consecutive_high_risk,
            }
        )

        if debug:
            print(f"  [COLLISION] risk={combined_risk:.2f} ({risk_level.name}), "
                  f"flow={flow_risk:.2f}, depth={depth_risk:.2f}, ttc={ttc:.1f}s")

        return state

    def get_safe_action(
        self,
        desired_action: int,
        state: CollisionState,
        debug: bool = False,
    ) -> Tuple[int, str]:
        """
        Get safe action given desired action and collision state.

        Implements the collision avoidance state machine:
        - CRITICAL: Emergency stop, backoff
        - HIGH: Stop, yaw scan to find open direction
        - MEDIUM: Slow down, prefer turning over forward
        - LOW/NONE: Allow desired action

        Args:
            desired_action: Action from high-level planner (0-4)
            state: Current collision state
            debug: Print debug info

        Returns:
            (safe_action, reason)
        """
        # Action mapping: 0=stay, 1=forward, 2=back, 3=left, 4=right

        if state.risk_level == RiskLevel.CRITICAL:
            # Emergency: rotate 180° to face backward, then check if clear
            # This avoids blindly backing into walls

            if self._backoff_rotation_frames < self._backoff_rotation_target:
                # Still rotating - continue turning
                self._backoff_rotation_frames += 1
                if debug:
                    print(f"  [COLLISION] CRITICAL - rotating to face backoff direction "
                          f"({self._backoff_rotation_frames}/{self._backoff_rotation_target})")
                return self._backoff_turn_direction, "backoff_rotation"
            else:
                # Finished rotating - now facing backward direction
                # Forward (action 1) is now the backoff direction
                # Check if it's clear via the state (which assesses forward risk)
                if state.combined_risk < self.thresholds['high']:
                    # Path is now clear, move forward (which is backing off)
                    if debug:
                        print(f"  [COLLISION] CRITICAL - backoff direction clear, moving")
                    # Reset rotation state for next time
                    self._backoff_rotation_frames = 0
                    return 1, "backoff_move_forward"
                else:
                    # Still blocked after rotation - try staying or turning more
                    if debug:
                        print(f"  [COLLISION] CRITICAL - backoff direction blocked, staying")
                    # Reset and try a different direction next time
                    self._backoff_rotation_frames = 0
                    # Alternate turn direction for next attempt
                    self._backoff_turn_direction = 3 if self._backoff_turn_direction == 4 else 4
                    return 0, "backoff_blocked_stay"

        elif state.risk_level == RiskLevel.HIGH:
            # High risk: stop forward motion, initiate yaw scan
            if desired_action == 1:  # Forward blocked
                # Pick turn direction based on last scan or alternate
                if state.escape_direction is not None:
                    action = state.escape_direction
                    reason = "escape_to_open"
                else:
                    action = 4 if self._consecutive_high_risk % 2 == 0 else 3
                    reason = "yaw_scan"

                if debug:
                    print(f"  [COLLISION] HIGH - turning instead of forward")
                return action, reason
            else:
                # Non-forward actions are usually OK
                return desired_action, "proceed_non_forward"

        elif state.risk_level == RiskLevel.MEDIUM:
            # Medium: allow turns, block sustained forward
            if desired_action == 1 and self._consecutive_high_risk > 2:
                return 0, "pause_forward"  # Stay in place
            return desired_action, "proceed_cautious"

        else:
            # LOW or NONE: proceed normally
            # Reset backoff rotation state when safe
            if self._backoff_rotation_frames > 0:
                self._backoff_rotation_frames = 0
            return desired_action, "proceed"

    def yaw_scan(
        self,
        frame: np.ndarray,
        yaw_angle: float,
        depth_map: np.ndarray = None,
    ) -> Optional[int]:
        """
        Perform yaw scan to find open direction.

        Call this repeatedly as the drone rotates. Once enough samples
        are collected, returns the best direction.

        Args:
            frame: Current frame
            yaw_angle: Current yaw angle (radians)
            depth_map: Optional depth map

        Returns:
            Best escape direction (3=left, 4=right) or None if still scanning
        """
        # Assess risk at current yaw
        state = self.assess_risk(frame, depth_map)
        self._yaw_scan_data.append((yaw_angle, state.combined_risk))

        # Need at least 5 samples spanning ~60 degrees
        if len(self._yaw_scan_data) < 5:
            return None

        # Find direction with lowest risk
        yaws = [y for y, r in self._yaw_scan_data]
        risks = [r for y, r in self._yaw_scan_data]

        yaw_span = max(yaws) - min(yaws)
        if yaw_span < 0.5:  # ~30 degrees
            return None  # Need more rotation

        # Find best yaw
        best_idx = np.argmin(risks)
        best_yaw = yaws[best_idx]
        current_yaw = yaw_angle

        # Determine turn direction
        yaw_diff = best_yaw - current_yaw
        if yaw_diff > 0:
            direction = 3  # Turn left to reach best yaw
        else:
            direction = 4  # Turn right

        # Clear scan data after decision
        self._yaw_scan_data = []

        return direction

    def reset(self):
        """Reset collision avoidance state."""
        if self.flow_ttc:
            self.flow_ttc.reset()
        self._consecutive_high_risk = 0
        self._escape_in_progress = False
        self._yaw_scan_data = []
        self._backoff_rotation_frames = 0
        self._backoff_turn_direction = 4


def create_collision_reflex(depth_estimator=None) -> CollisionAvoidance:
    """
    Create a ready-to-use collision avoidance instance.

    Args:
        depth_estimator: Optional DepthEstimator for hybrid mode

    Returns:
        Configured CollisionAvoidance instance
    """
    return CollisionAvoidance(
        depth_estimator=depth_estimator,
        use_flow=True,
        use_depth=depth_estimator is not None,
        flow_weight=0.6,  # Slightly favor flow (faster response)
    )


# Test code
if __name__ == "__main__":
    print("Testing Collision Avoidance...")

    # Test with webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam available")
        exit(1)

    collision = CollisionAvoidance(
        use_flow=True,
        use_depth=False,  # No depth estimator in test
    )

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        state = collision.assess_risk(frame, debug=True)

        # Show flow visualization
        if state.debug_info and state.debug_info.get('flow_vis') is not None:
            flow_vis = state.debug_info['flow_vis']
            combined = np.hstack([frame, flow_vis])
        else:
            combined = frame

        # Draw risk indicator
        color = {
            RiskLevel.NONE: (0, 255, 0),
            RiskLevel.LOW: (0, 255, 255),
            RiskLevel.MEDIUM: (0, 165, 255),
            RiskLevel.HIGH: (0, 0, 255),
            RiskLevel.CRITICAL: (0, 0, 255),
        }[state.risk_level]

        cv2.putText(combined, f"Risk: {state.risk_level.name} ({state.combined_risk:.2f})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(combined, f"TTC: {state.ttc_estimate:.1f}s",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Collision Avoidance", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
