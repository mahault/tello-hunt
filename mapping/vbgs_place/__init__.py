"""
VBGS Local Place Models.

Uses Variational Bayesian Gaussian Splatting to model
the appearance of each place/location without requiring
accurate metric poses.

Provides:
- VBGSPlaceModel: Local Gaussian mixture for one place
- KeyframeSelector: Decides when to add new keyframes
- PlaceManager: Manages all place models
"""

from .place_model import VBGSPlaceModel, SimplePlaceModel
from .keyframe_selector import KeyframeSelector, AdaptiveKeyframeSelector
from .place_manager import PlaceManager

__all__ = [
    'VBGSPlaceModel',
    'SimplePlaceModel',
    'KeyframeSelector',
    'AdaptiveKeyframeSelector',
    'PlaceManager',
]
