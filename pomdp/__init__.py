"""
POMDP-based world models for Tello drone navigation.

Three layered POMDPs:
1. World Model - Learned topological map for localization
2. Human Search - Belief over person locations
3. Interaction Mode - Action selection via Expected Free Energy
"""

from .config import *
from .observation_encoder import (
    ObservationToken,
    encode_yolo_detections,
    create_empty_observation,
    observation_to_text,
)
from .topological_map import TopologicalMap, PlaceNode, EdgeStats
from .similarity import (
    cosine_similarity,
    weighted_cosine_similarity,
    is_same_location,
    find_most_similar,
    batch_cosine_similarity,
)
from .map_persistence import (
    save_map,
    load_map,
    load_latest_map,
    list_saved_maps,
)
from .world_model import WorldModel, LocalizationResult
from .human_search import HumanSearchPOMDP, HumanSearchResult
from .interaction_mode import (
    InteractionModePOMDP,
    InteractionResult,
    action_to_rc_control,
    SEARCHING,
    APPROACHING,
    INTERACTING,
    DISENGAGING,
    ACTION_CONTINUE_SEARCH,
    ACTION_APPROACH,
    ACTION_INTERACT_LED,
    ACTION_INTERACT_WIGGLE,
    ACTION_BACKOFF,
    ACTION_LAND,
)
from .exploration_mode import (
    ExplorationModePOMDP,
    ExplorationResult,
    exploration_action_to_rc_control,
    SCANNING,
    APPROACHING_FRONTIER,
    BACKTRACKING,
    TRANSITIONING,
)
from .image_encoder import (
    ImageEncoder,
    encode_frame,
    embedding_similarity,
    find_most_similar_embedding,
    CLIP_EMBEDDING_DIM,
)
