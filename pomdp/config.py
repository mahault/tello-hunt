"""
Configuration constants for POMDP world models.

Defines state spaces, observation spaces, and thresholds.
"""

# =============================================================================
# Topological Map Configuration
# =============================================================================

# Maximum number of locations in the learned map
N_MAX_LOCATIONS = 50

# Similarity threshold for localization (cosine similarity)
# Above this = same location, below = new location
LOCATION_SIMILARITY_THRESHOLD = 0.7

# Minimum observations before a location is considered "established"
MIN_VISITS_FOR_ESTABLISHED = 3

# =============================================================================
# YOLO Object Types for Room Detection
# =============================================================================

# COCO class IDs we track for room/location inference
# Maps COCO class ID -> our internal type index
COCO_TO_TYPE = {
    0: 0,    # person
    56: 1,   # chair
    57: 2,   # couch
    58: 3,   # potted plant
    59: 4,   # bed
    60: 5,   # dining table
    61: 6,   # toilet
    62: 7,   # tv
    63: 8,   # laptop
    67: 9,   # cell phone
    68: 10,  # oven
    70: 11,  # toaster
    71: 12,  # sink
    72: 13,  # refrigerator
    73: 14,  # book
    74: 15,  # clock
    79: 16,  # toothbrush
}

# Reverse mapping: type index -> name
TYPE_NAMES = [
    'person', 'chair', 'couch', 'potted_plant', 'bed',
    'dining_table', 'toilet', 'tv', 'laptop', 'cell_phone',
    'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'toothbrush'
]

N_OBJECT_TYPES = len(TYPE_NAMES)  # 17

# Observation levels for discretization
OBS_LEVELS = ['absent', 'low_conf', 'high_conf']
N_OBS_LEVELS = 3

# Confidence thresholds for discretization
CONF_THRESHOLD_LOW = 0.3
CONF_THRESHOLD_HIGH = 0.6

# =============================================================================
# Human Search POMDP
# =============================================================================

# Person observation categories
PERSON_OBS = ['not_detected', 'detected_left', 'detected_center',
              'detected_right', 'detected_close']
N_PERSON_OBS = 5

# Position thresholds for person detection discretization
PERSON_LEFT_THRESHOLD = -0.3   # cx < this = left
PERSON_RIGHT_THRESHOLD = 0.3   # cx > this = right
PERSON_CLOSE_THRESHOLD = 0.5   # area > this = close

# =============================================================================
# Interaction Mode POMDP
# =============================================================================

# Engagement states
ENGAGEMENT_STATES = ['searching', 'approaching', 'interacting', 'disengaging']
N_ENGAGEMENT_STATES = 4

# Actions available to the agent
ACTIONS = [
    'continue_search',  # Keep rotating/exploring
    'approach',         # Move toward detected person
    'interact_led',     # Signal with LED (if available)
    'interact_wiggle',  # Wiggle motion for attention
    'backoff',          # Increase distance
    'land'              # Safe landing
]
N_ACTIONS = 6

# Movement actions for B matrix (transitions between locations)
MOVEMENT_ACTIONS = ['stay', 'forward', 'back', 'left', 'right']
N_MOVEMENT_ACTIONS = 5

# =============================================================================
# Drone Control Parameters
# =============================================================================

# Target person area in frame (for approach behavior)
PERSON_TARGET_AREA = 0.80
PERSON_TOO_CLOSE = 0.90

# RC control limits
MAX_FB = 20   # Forward/back velocity
MAX_YAW = 25  # Rotation rate
MAX_UD = 0    # Up/down (disabled for safety)

# Search behavior
SEARCH_YAW = 35  # Rotation speed during search

# =============================================================================
# Learning Parameters (Dirichlet priors)
# =============================================================================

# Prior concentration for Dirichlet-Categorical learning
# Higher = slower learning, more stable
# Lower = faster learning, more reactive
DIRICHLET_PRIOR_ALPHA = 1.0

# Learning rate for transition updates
TRANSITION_LEARNING_RATE = 0.1

# Decay factor for old observations (optional forgetting)
OBSERVATION_DECAY = 0.99

# =============================================================================
# Active Inference Parameters
# =============================================================================

# Temperature for action selection softmax
# Lower = more deterministic, higher = more exploratory
ACTION_TEMPERATURE = 0.5

# Epistemic value weight (curiosity/exploration)
EPISTEMIC_WEIGHT = 1.0

# Pragmatic value weight (goal achievement)
PRAGMATIC_WEIGHT = 1.0
