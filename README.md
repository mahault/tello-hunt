# Tello Hunt

Autonomous drone tracking and exploration using DJI Tello/Tello Talent with YOLO object detection and Active Inference (POMDP-based decision making).

## Features

### Core Capabilities
- **Autonomous Exploration**: Frontier-based exploration with topological mapping
- **Person Tracking**: Search, approach, signal, and backoff behavior
- **Cat Shadowing**: Follow cats while avoiding people (safety-first design)
- **3D Simulation**: Test algorithms without physical drone using GLB house models

### Technical Stack
- **JAX JIT Compilation**: Real-time POMDP inference
- **Three-Layer POMDP**: World Model + Human Search + Interaction Mode
- **ORB Place Recognition**: 87.5% accuracy in realistic environments
- **Hybrid Collision Avoidance**: Optical flow + monocular depth
- **Occupancy Grid Mapping**: Depth-based spatial mapping with free-space carving

## Requirements

- Python 3.10+
- DJI Tello or Tello Talent drone (for real flight)
- CUDA-capable GPU (recommended for depth estimation)

## Installation

```bash
# Create environment
conda env create -f environment.yml
conda activate tello-hunt

# Or with pip
pip install opencv-python djitellopy ultralytics keyboard jax transformers

# Download YOLO weights (run once)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Quick Start

### Option 1: Simulation (No Drone Required)

Test the full exploration system in 3D simulation:

```bash
# Frontier-based exploration (recommended)
python test_full_pipeline.py --frontier

# Topological exploration
python test_full_pipeline.py --topological

# Semantic exploration with room priors
python test_full_pipeline.py --semantic
```

### Option 2: Real Drone

#### Step 1: Connect to Drone
1. Power on the drone
2. Connect your computer to the drone's WiFi (TELLO-XXXXXX)

#### Step 2: Test Connection
```bash
# Test commands
python cmd_debug.py

# Test video stream
python video_debug_pyav.py
```

#### Step 3: Run Applications
```bash
# Person tracking with POMDP
python person_hunter_pomdp.py

# Simple person tracking
python person_hunter_safe.py

# Cat shadowing with safety
python cat_safe_shadow.py
```

## Project Structure

```
tello-hunt/
├── pomdp/                    # POMDP inference engine (~2.5K lines)
│   ├── core.py              # JAX JIT belief updates, EFE
│   ├── world_model.py       # Semantic localization
│   ├── frontier_explorer.py # Frontier-based navigation
│   ├── human_search.py      # Person location tracking
│   └── interaction_mode.py  # Action selection via Active Inference
│
├── mapping/                  # Place recognition & mapping
│   ├── cscg/                # Clone-Structured Cognitive Graphs
│   └── semantic_world_model.py
│
├── simulator/                # 3D testing environment
│   ├── glb_simulator.py     # GLB model renderer (pyrender)
│   └── simple_3d.py         # Lightweight raycasting sim
│
├── safety/                   # Safety monitoring (outside POMDP)
│   └── overrides.py         # Battery, collision detection
│
├── utils/                    # Utilities
│   ├── occupancy_map.py     # Spatial grid mapping
│   ├── collision_avoidance.py
│   └── depth_estimator.py
│
├── person_hunter_pomdp.py   # Full POMDP hunting
├── person_hunter_safe.py    # Simple tracking
└── test_full_pipeline.py    # Simulation testing
```

## Exploration Modes

### Frontier-Based (Recommended)
```bash
python test_full_pipeline.py --frontier
```
- Standard robotics approach: explore boundaries between known/unknown space
- A* pathfinding to frontier targets
- Escape strategies for stuck situations
- Blocked-edge memory to avoid repeated failures

### Topological
```bash
python test_full_pipeline.py --topological
```
- Pure place-graph navigation
- BFS to find nearest unexplored transitions
- Good for structured environments

### Semantic (Experimental)
```bash
python test_full_pipeline.py --semantic
```
- EFE-based exploration with room/object priors
- Seeks specific room types (kitchen, bedroom, etc.)
- YOLO object detection for room classification

## Controls

| Mode | Key | Action |
|------|-----|--------|
| All | Q/ESC | Quit |
| All | SPACE | Toggle auto/manual |
| All | R | Reset |
| Manual | W/Up | Forward |
| Manual | S/Down | Backward |
| Manual | A/Left | Turn left |
| Manual | D/Right | Turn right |

## Configuration

### POMDP Settings (`pomdp/config.py`)
- `N_MAX_LOCATIONS = 50` - Maximum learned places
- `LOCATION_SIMILARITY_THRESHOLD = 0.5` - Place matching
- `ACTION_TEMPERATURE = 2.0` - Exploration vs exploitation

### Safety Settings (`safety/overrides.py`)
- Battery warning: 20%
- Battery critical: 10%
- Collision detection: 5 consecutive blocked frames

### Occupancy Mapping (`utils/occupancy_map.py`)
- Resolution: 10cm per cell
- Depth obstacles: value 60 (low confidence)
- Collision obstacles: value 0 (locked, high confidence)

## Troubleshooting

### Drone Issues
1. **Not responding**: Check WiFi connection to TELLO-XXXXXX
2. **Video not working**: Ensure port 11111 isn't blocked
3. **Erratic behavior**: Check battery level (won't fly < 10%)

### Simulation Issues
1. **GLB not loading**: Install `pip install pyrender trimesh`
2. **Slow rendering**: Reduce window size or disable depth estimation
3. **Stuck in corners**: Escape logic triggers after 5 consecutive blocks

### Exploration Issues
1. **Not finding doors**: Check occupancy map for false obstacles
2. **Oscillating at boundaries**: Blocked-edge memory should prevent this
3. **Never leaving room**: Verify doorway regions aren't painted as walls

## Technical Details

### Three-Layer POMDP Architecture

```
YOLO Frame → ObservationToken → WorldModel → [Exploration | Hunting] → RC Control
                                    ↓
                             Safety Monitor (highest priority)
```

1. **World Model**: Soft belief over 50 learned locations
2. **Human Search**: Belief over person locations
3. **Interaction Mode**: EFE-based action selection

### Occupancy Grid Mapping

Two types of obstacles with different confidence:
- **Depth-sensed** (value ~60): Single cell, can be overwritten
- **Collision-confirmed** (value 0): 3x3 dilation, locked

This prevents doorways from being "painted over" by depth sensing at angles.

### Escape Strategies

When stuck (6+ consecutive blocks):
1. **Short-term (1-5 blocks)**: Alternate left/right turns
2. **Medium-term (6-11 blocks)**: 360° scan for clear direction
3. **Long-term (12+ blocks)**: Blacklist target, select new goal

### Room Transition Tracking

The frontier explorer tracks room-to-room transitions as first-class events:
- Door crossings counted with timestamps
- Unique doorways used tracked
- Room visit history maintained

Doorway detection uses aspect ratio heuristics (elongated frontier clusters).
Doorway frontiers receive a 0.4m distance bonus for prioritization.

## Safety

- Fly in open spaces away from people and obstacles
- Keep line of sight with the drone
- All scripts auto-land on exit or error
- Speed limits are conservative by default
- Safety monitor runs independently of POMDP

## References

- [Bio-Inspired Topological Navigation](https://arxiv.org/html/2508.07267) - Active Inference approach
- [Clone-Structured Cognitive Graphs](https://github.com/vicariousinc/naturecomm_cscg) - Perceptual aliasing
- [pymdp](https://github.com/infer-actively/pymdp) - Active inference library
