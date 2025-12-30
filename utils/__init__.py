"""Utility modules for Tello drone control."""

from .frame_grabber import FrameGrabber, clamp

# Optional depth estimation (requires transformers)
try:
    from .depth_estimator import DepthEstimator
except ImportError:
    DepthEstimator = None

# Visual odometry and spatial mapping
from .visual_odometry import MonocularVO, ActionBasedOdometry, OdometryState
from .occupancy_map import OccupancyMap, SpatialMapper, MapConfig

# Collision avoidance
from .collision_avoidance import (
    CollisionAvoidance,
    CollisionState,
    RiskLevel,
    OpticalFlowTTC,
    create_collision_reflex,
)
