"""
Monocular Visual Odometry for Tello Drone.

Tracks camera motion by matching ORB features between consecutive frames.
Estimates rotation and translation to build a trajectory.

This is realistic for the Tello - works with just the RGB camera.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class OdometryState:
    """Current odometry state."""
    x: float           # Position in meters (forward)
    y: float           # Position in meters (right)
    z: float           # Position in meters (up)
    yaw: float         # Heading in radians
    pitch: float       # Pitch in radians
    roll: float        # Roll in radians

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.yaw, self.pitch, self.roll])

    def copy(self) -> 'OdometryState':
        return OdometryState(self.x, self.y, self.z, self.yaw, self.pitch, self.roll)


class MonocularVO:
    """
    Monocular Visual Odometry using ORB features.

    Tracks camera motion between frames by:
    1. Detecting ORB keypoints
    2. Matching to previous frame
    3. Estimating Essential matrix
    4. Decomposing to R, t
    5. Accumulating pose

    Note: Monocular VO has scale ambiguity - we estimate relative motion
    and apply a scale factor based on expected drone speed.
    """

    def __init__(
        self,
        focal_length: float = 500.0,  # Approximate for Tello camera
        cx: float = 320.0,            # Principal point x
        cy: float = 240.0,            # Principal point y
        scale_factor: float = 0.03,   # Meters per unit (tune based on drone speed)
        min_matches: int = 20,        # Minimum matches for pose estimation
    ):
        """
        Initialize visual odometry.

        Args:
            focal_length: Camera focal length in pixels
            cx, cy: Principal point (image center)
            scale_factor: Scale to convert to metric (monocular has no scale)
            min_matches: Minimum feature matches to estimate motion
        """
        self.focal_length = focal_length
        self.cx = cx
        self.cy = cy
        self.scale_factor = scale_factor
        self.min_matches = min_matches

        # Camera matrix
        self.K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # ORB detector
        self.orb = cv2.ORB_create(nfeatures=1000)

        # Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State
        self.pose = OdometryState(0, 0, 0, 0, 0, 0)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None

        # Trajectory history
        self.trajectory: List[Tuple[float, float, float]] = [(0, 0, 0)]

        # For visualization
        self.last_matches = 0
        self.last_inliers = 0

    def update(self, frame: np.ndarray) -> OdometryState:
        """
        Update odometry with new frame.

        Args:
            frame: BGR image from camera

        Returns:
            Updated odometry state
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB features
        kp, des = self.orb.detectAndCompute(gray, None)

        if des is None or len(kp) < self.min_matches:
            # Not enough features
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.pose

        if self.prev_des is None:
            # First frame
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.pose

        # Match features
        matches = self.bf.knnMatch(self.prev_des, des, k=2)

        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        self.last_matches = len(good_matches)

        if len(good_matches) < self.min_matches:
            # Not enough good matches
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.pose

        # Extract matched points
        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])

        # Estimate Essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.pose

        self.last_inliers = int(mask.sum()) if mask is not None else 0

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        # Convert rotation matrix to Euler angles
        yaw, pitch, roll = self._rotation_to_euler(R)

        # Apply scale factor (monocular has no absolute scale)
        t_scaled = t.flatten() * self.scale_factor

        # Transform translation to world frame using current heading
        cos_yaw = np.cos(self.pose.yaw)
        sin_yaw = np.sin(self.pose.yaw)

        # t is in camera frame: x=right, y=down, z=forward
        # Convert to world frame: x=forward, y=right, z=up
        dx_world = t_scaled[2] * cos_yaw - t_scaled[0] * sin_yaw  # forward
        dy_world = t_scaled[2] * sin_yaw + t_scaled[0] * cos_yaw  # right
        dz_world = -t_scaled[1]  # up

        # Update pose
        self.pose.x += dx_world
        self.pose.y += dy_world
        self.pose.z += dz_world
        self.pose.yaw += yaw * 0.1  # Dampen rotation updates

        # Store trajectory
        self.trajectory.append((self.pose.x, self.pose.y, self.pose.z))

        # Keep trajectory bounded
        if len(self.trajectory) > 10000:
            self.trajectory = self.trajectory[-5000:]

        # Update previous frame
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des

        return self.pose

    def _rotation_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (yaw, pitch, roll)."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

        if sy > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return yaw, pitch, roll

    def update_from_action(self, action: int, moved: bool):
        """
        Update pose based on commanded action (dead reckoning fallback).

        Used when visual odometry fails or as a prior.

        Args:
            action: 0=stay, 1=forward, 2=back, 3=left, 4=right
            moved: Whether the action succeeded
        """
        if not moved:
            return

        # Approximate motion per action (tune based on drone speed)
        move_dist = 0.3  # meters
        turn_angle = np.radians(15)  # degrees

        if action == 1:  # forward
            self.pose.x += move_dist * np.cos(self.pose.yaw)
            self.pose.y += move_dist * np.sin(self.pose.yaw)
        elif action == 2:  # backward
            self.pose.x -= move_dist * np.cos(self.pose.yaw)
            self.pose.y -= move_dist * np.sin(self.pose.yaw)
        elif action == 3:  # turn left
            self.pose.yaw += turn_angle
        elif action == 4:  # turn right
            self.pose.yaw -= turn_angle

        # Normalize yaw
        self.pose.yaw = (self.pose.yaw + np.pi) % (2 * np.pi) - np.pi

        # Update trajectory
        self.trajectory.append((self.pose.x, self.pose.y, self.pose.z))

    def get_trajectory_2d(self) -> np.ndarray:
        """Get 2D trajectory (x, y) as numpy array."""
        return np.array([(p[0], p[1]) for p in self.trajectory])

    def reset(self):
        """Reset odometry to origin."""
        self.pose = OdometryState(0, 0, 0, 0, 0, 0)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.trajectory = [(0, 0, 0)]

    def get_status(self) -> str:
        """Get status string for display."""
        return f"VO: ({self.pose.x:.2f}, {self.pose.y:.2f}) yaw={np.degrees(self.pose.yaw):.0f}° matches={self.last_matches}"


class ActionBasedOdometry:
    """
    Simple odometry based on commanded actions.

    More reliable than visual odometry for indoor navigation
    when actions have predictable effects.
    """

    def __init__(
        self,
        move_distance: float = 0.5,   # Meters per forward/back action
        turn_angle: float = 15.0,     # Degrees per turn action
    ):
        self.move_distance = move_distance
        self.turn_angle = np.radians(turn_angle)

        self.pose = OdometryState(0, 0, 0, 0, 0, 0)
        self.trajectory: List[Tuple[float, float]] = [(0, 0)]

    def update(self, action: int, moved: bool) -> OdometryState:
        """
        Update pose based on action.

        Args:
            action: 0=stay, 1=forward, 2=back, 3=left, 4=right
            moved: Whether action succeeded (False if blocked)

        Returns:
            Updated pose
        """
        if action == 3:  # turn left (CCW in odometry coords where yaw=0 is +X)
            self.pose.yaw += self.turn_angle
        elif action == 4:  # turn right (CW)
            self.pose.yaw -= self.turn_angle
        elif moved:  # Only update position if movement succeeded
            if action == 1:  # forward
                self.pose.x += self.move_distance * np.cos(self.pose.yaw)
                self.pose.y += self.move_distance * np.sin(self.pose.yaw)
            elif action == 2:  # backward
                self.pose.x -= self.move_distance * np.cos(self.pose.yaw)
                self.pose.y -= self.move_distance * np.sin(self.pose.yaw)

        # Normalize yaw to [-pi, pi]
        self.pose.yaw = (self.pose.yaw + np.pi) % (2 * np.pi) - np.pi

        # Record trajectory
        self.trajectory.append((self.pose.x, self.pose.y))

        return self.pose

    def get_trajectory_2d(self) -> np.ndarray:
        """Get 2D trajectory as numpy array."""
        return np.array(self.trajectory)

    def reset(self):
        """Reset to origin."""
        self.pose = OdometryState(0, 0, 0, 0, 0, 0)
        self.trajectory = [(0, 0)]
