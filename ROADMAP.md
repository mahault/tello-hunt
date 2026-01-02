# POMDP World Model Implementation Plan

## Overview

Add three layered POMDPs to tello-hunt for intelligent room-aware person hunting:

1. **World Model POMDP** - "Which room/zone am I in?" (semantic localization)
2. **Human Search POMDP** - "Where are humans likely to be?" (belief tracking)
3. **Interaction Mode POMDP** - "What action should I take?" (policy selection)

All implemented in JAX with JIT compilation for real-time performance.

---

## Project Structure

```
pomdp/
├── __init__.py
├── config.py              # Constants, thresholds
├── core.py                # JAX JIT belief update functions
├── observation_encoder.py # YOLO → fixed observation tokens
├── topological_map.py     # TopologicalMap, LocationNode classes
├── similarity.py          # Observation similarity metrics
├── world_model.py         # Location belief + learning
├── human_search.py        # Person location belief
├── interaction_mode.py    # Action selection via EFE
├── map_persistence.py     # Save/load learned environment maps
└── priors.py              # Context-based priors

safety/
├── __init__.py
└── overrides.py           # Battery, collision (outside POMDP)

utils/
└── frame_grabber.py       # Extracted from existing code

maps/                      # Saved learned environment maps
└── (learned_map_*.json)   # Persisted A, B counts + location graph

person_hunter_pomdp.py     # New main script
```

---

## State Spaces

### World Model - LEARNED TOPOLOGICAL MAP

Based on [Bio-Inspired Topological Navigation with Active Inference](https://arxiv.org/html/2508.07267):

**Topological Graph Structure:**
- **Nodes** = locations (stored observation signatures + object detections)
- **Edges** = transitions between locations (learned from movement)
- Graph grows incrementally as drone explores

**Localization (inferring current location):**
1. Capture current observation (YOLO detections → object signature vector)
2. Compare to all stored location signatures using similarity metric
3. If similarity > threshold → localized to that node
4. If no match above threshold → **new location discovered**, add node

**Similarity Metric Options:**
- Cosine similarity of object detection histograms
- Weighted by object confidence scores
- Could also use image features (but YOLO objects are more semantic)

**Map Expansion (EFE-guided):**
- Expected Free Energy guides exploration
- Prefer actions leading to uncertain/unexplored areas (epistemic value)
- Balance exploration vs exploitation (finding humans)

**Online Learning (Dirichlet pseudo-counts):**
- **A matrix**: P(observation|location) - updated when localized
  - `A_counts[obs, loc] += 1` each time obs seen at loc
- **B matrix**: P(location'|location, action) - updated on transitions
  - `B_counts[to_loc, from_loc, action] += 1` each transition
- Matrices derived: `A = normalize(A_counts + prior_α)`

**Handling Novelty/Dynamics:**
- Low confidence localization → explore to re-localize
- Observation mismatch at known location → environment changed
- Failed transitions → update B to mark blocked paths

### Human Search (N_LOCATIONS + 1 states)
- N learned locations + 1 "not visible" state
- Prior updated based on where humans were previously seen

### Interaction Mode (4 states)
- searching, approaching, interacting, disengaging

---

## Observation Encoding

YOLO detections → fixed-size tokens for JIT compatibility:

| Object | COCO ID | Room Indicator |
|--------|---------|----------------|
| oven | 68 | Kitchen |
| refrigerator | 72 | Kitchen |
| sink | 71 | Kitchen/Bathroom |
| couch | 57 | Living Room |
| tv | 62 | Living Room |
| bed | 59 | Bedroom |
| toilet | 61 | Bathroom |
| person | 0 | Human location |

Each object encoded as: `[detected, count, max_conf, avg_x, avg_y, avg_area]`
Discretized to 3 levels: absent, low_conf, high_conf

---

## Matrix Shapes (PyMDP-style A, B, C, D)

With N_LOCATIONS = 20 max learned places:

| POMDP | A (likelihood) | B (transition) | C (preference) | D (prior) |
|-------|----------------|----------------|----------------|-----------|
| World | (17, 3, 20) | (20, 20, 5) | (17, 3) | (20,) |
| Human | (5, 21, 20) | (21, 21) | (5,) | (21,) |
| Interaction | (5, 4) | (4, 4, 6) | (5,) | (4,) |

**Learning Parameters (Dirichlet counts):**
- `A_counts`: (17, 3, 20) - observation counts per location
- `B_counts`: (20, 20, 5) - transition counts per action
- `human_seen_counts`: (21,) - where humans were observed

Matrices derived from counts: `A = normalize(A_counts + prior)`

---

## Actions

| Action | RC Control | When |
|--------|------------|------|
| `continue_search` | (0, 0, 0, SEARCH_YAW) | No person, exploring |
| `approach` | (0, fb, 0, yaw) | Person detected, not close |
| `interact_wiggle` | (0, 0, 0, ±15) | Person at target distance |
| `interact_led` | LED pattern | Person engaged |
| `backoff` | (0, -MAX_FB, 5, 0) | Too close or collision |
| `land` | Landing sequence | Task complete or safety |

Actions selected by minimizing Expected Free Energy (active inference).

---

## Main Loop Integration

```python
while running:
    img = grabber.get_frame()

    # 1. Safety check (outside POMDP - hard overrides)
    safety = check_safety_overrides(tello)
    if safety != OK: handle_safety(...)

    # 2. YOLO detection
    res = model(img, imgsz=320)[0]

    # 3. Encode to fixed observation token
    detection = encode_yolo_detections(res.boxes, ...)

    # 4. POMDP update cycle (all JIT compiled)
    result = pomdp.update(detection)

    # 5. Execute action
    lr, fb, ud, yaw = action_to_rc_control(result['action'], ...)
    tello.send_rc_control(lr, fb, ud, yaw)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure ✓
- [x] `pomdp/config.py` - state/observation definitions, N_LOCATIONS constant
- [x] `pomdp/core.py` - JAX JIT functions:
  - `normalize()`, `softmax()`, `entropy()`, `kl_divergence()`
  - `belief_update()`, `belief_update_from_A()`, `belief_update_multi_modality()`
  - **VFE functions (for perception):**
    - `accuracy()` - E_q[log p(o|s)], how well belief explains observation
    - `complexity()` - KL[q||p], how much belief diverged from prior
    - `variational_free_energy()` - complexity - accuracy
    - `surprisal()` - -log p(o), novelty detection
    - `vfe_components()` - all VFE metrics at once
    - `is_novel_observation()` - threshold-based novelty check
    - `belief_update_with_vfe()` - update + VFE monitoring
  - **EFE functions (for action):**
    - `predict_next_belief()`, `predict_observation()`
    - `expected_free_energy()`, `compute_all_efe()`
    - `select_action()`
  - **Learning:**
    - `update_dirichlet_counts()`, `counts_to_distribution()`
  - **Dynamic state spaces:**
    - `expand_belief()`, `expand_A_matrix()`, `expand_B_matrix()`
- [x] `pomdp/observation_encoder.py` - YOLO → fixed tokens
  - `ObservationToken` dataclass
  - `encode_yolo_detections()` - convert YOLO boxes to fixed-size token
  - `discretize_person_obs()` - categorical person observation

### Phase 2: Topological Map Module ✓
- [x] `pomdp/topological_map.py`:
  - `TopologicalMap` class:
    - `nodes`: list of LocationNode (observation signature, visit count)
    - `edges`: list of Edge with transition counts per action
    - `add_node(observation)` - create new location
    - `find_best_match(observation)` - localization via similarity
    - `localize(observation)` - find/create location with threshold
    - `add_edge(from_node, to_node, action)` - record transition
    - `get_A_matrix()` - generate observation likelihood from counts
    - `get_B_matrix()` - generate transition probabilities from edges
  - `LocationNode` dataclass:
    - `id`: unique identifier
    - `observation_signature`: running mean of object vectors
    - `A_counts`: observation counts at this location (N_OBJECT_TYPES, N_OBS_LEVELS)
    - `visit_count`: times visited
    - `update_signature()` - online mean update
    - `update_A_counts()` - increment observation counts
  - `Edge` dataclass for transitions with counts
- [x] `pomdp/similarity.py`:
  - `cosine_similarity()` - cosine similarity of vectors
  - `weighted_cosine_similarity()` - with object importance weights
  - `jaccard_similarity()` - set-based similarity
  - `is_same_location(obs, node, threshold)` - localization decision
  - `batch_cosine_similarity()` (JIT) - fast similarity to all locations
  - `find_best_match_jit()` (JIT) - fast localization
  - `DEFAULT_OBJECT_WEIGHTS` - distinctive object weighting
- [x] `pomdp/map_persistence.py`:
  - `save_map(path, topo_map)` - serialize graph + counts to JSON
  - `load_map(path)` - restore learned environment
  - `load_latest_map()` - load most recent map
  - `list_saved_maps()` - enumerate saved maps
  - `export_map_summary()` - human-readable map description

### Phase 3: World Model POMDP (with topological map) ✓
- [x] `pomdp/world_model.py`:
  - `WorldModel` class - main POMDP for semantic localization
  - `localize(observation, action)` - update belief, detect new locations
  - Soft belief over all locations (not hard assignment)
  - VFE-based novelty detection for new location discovery
  - Automatic A/B matrix learning from experience
  - `get_exploration_target()` - EFE-guided action recommendations
  - `get_location_info()` - query location details
  - `get_belief_entropy()` - uncertainty measure
  - Save/load via `WorldModel.save()` and `WorldModel.load()`
  - `LocalizationResult` dataclass with VFE diagnostics

### Phase 4: Human Search POMDP ✓
- [x] `pomdp/human_search.py`:
  - `HumanSearchPOMDP` class - belief tracking over human locations
  - State space: N_locations + 1 ("not visible")
  - Dynamic A matrix: P(person_obs | human_loc, drone_loc)
  - B matrix: Human transition model (mostly stationary)
  - `update()` - Bayesian belief update from person observations
  - `_update_sighting_counts()` - learn where humans appear
  - `_get_search_target()` - recommend where to look next
  - `expand_to_locations()` - dynamic state space expansion
  - `HumanSearchResult` dataclass with belief diagnostics
  - Save/load via `save_state()` and `load_state()`

### Phase 5: Interaction Mode POMDP ✓
- [x] `pomdp/interaction_mode.py`:
  - `InteractionModePOMDP` class - action selection via active inference
  - State space: 4 engagement states (searching, approaching, interacting, disengaging)
  - Action space: 6 actions (continue_search, approach, interact_led, interact_wiggle, backoff, land)
  - Hand-designed A matrix: P(person_obs | engagement_state)
  - Hand-designed B matrix: P(state' | state, action) - action effects on engagement
  - C preferences: favor detecting people, especially close
  - `update()` - belief update + EFE-based action selection
  - `update_with_action_override()` - for safety overrides
  - `get_action_for_state()` - best action to reach target state
  - `action_to_rc_control()` - convert action to drone RC commands
  - `InteractionResult` dataclass with belief, action, EFE diagnostics
  - Save/load via `save_state()` and `load_state()`

### Phase 6: Safety Module ✓
- [x] `safety/__init__.py` - Package exports
- [x] `safety/overrides.py`:
  - `SafetyState` dataclass: battery level, warning/critical flags, contact detection, emergency state
  - `SafetyOverride` constants: NONE, BACKOFF, HOVER, LAND, EMERGENCY_LAND
  - `SafetyMonitor` class:
    - `__init__(tello)` - stores Tello reference
    - `update(commanded_fb)` - main safety check, returns (SafetyState, override_code)
    - `_check_battery()` - rate-limited battery query (every 5s)
    - `_check_contact(commanded_fb)` - blocked movement detection (5 frames threshold)
    - `get_override_rc(override)` - RC values for override actions
    - `reset_contact()` - reset after backoff
    - `should_land()` - check if landing required
  - Battery thresholds: warn at 20%, critical at 10%
  - Contact detection: speed < 5 cm/s while commanding forward for 5 frames

### Phase 7: Integration ✓
- [x] `utils/frame_grabber.py` - extracted from person_hunter_safe.py
  - `FrameGrabber` class: threaded frame capture with lock
  - `get_frame()` - returns thread-safe frame copy
  - `clamp()` utility function
- [x] `utils/__init__.py` - exports FrameGrabber, clamp
- [x] `person_hunter_pomdp.py` - full POMDP integration script
  - `POMDPController` class:
    - `__init__(load_existing_map)` - initializes all 3 POMDPs
    - `update(yolo_boxes, ...)` - full update cycle with safety override support
    - `save_map(name)` - persist learned world model
    - `get_diagnostics()` - status from all POMDPs
  - Main loop structure:
    1. Safety check FIRST (before POMDP)
    2. YOLO detection
    3. POMDP update cycle (WorldModel → HumanSearch → InteractionMode)
    4. Execute RC control
    5. Visualization overlay
    6. Exit condition checks
  - `draw_pomdp_overlay()` - visualization with:
    - Battery indicator (color-coded)
    - Location + confidence
    - Engagement state + action
    - Human belief bar
    - Safety warnings (contact, low battery)
    - RC control indicator
  - Pre-flight warmup loop
  - Graceful cleanup and map save on exit

### Phase 8: Exploration Mode ✓ (REWRITTEN Dec 2025)

**Previous approach (deprecated → `legacy/exploration_mode_old.py`):**
- Complex door-seeking with depth scanning
- VFE-based state machine
- Struggled to find doors and exit rooms

**New approach: Topological Frontier-Based Exploration**

Uses a graph of places and connections for systematic exploration:
1. **Places** = ORB keyframe nodes (from CSCG place recognition)
2. **Edges** = Connections discovered when moving between places
3. **Frontier** = Places with untried movement directions

**Exploration strategy:**
```
if current_place has untried directions:
    try an untried direction (prefer forward > left/right > backward)
else:
    BFS to find nearest frontier node
    backtrack along known edges to reach it
```

**Why this works:**
- Naturally handles finding doors (untried directions that lead to new places)
- Never gets stuck (always have a frontier to explore or exploration is complete)
- Efficient coverage (BFS ensures we explore nearby frontiers first)
- Clean graph structure enables backtracking via known paths

- [x] `pomdp/exploration_mode.py` (rewritten):
  - `ExplorationGraph` class - topological graph tracking:
    - `edges[place][action] = destination` - known transitions
    - `tried[place] = {actions}` - actions attempted
    - `blocked[place] = {actions}` - walls detected
    - `find_nearest_frontier(place)` - BFS to frontier
    - `get_path_to(from, to)` - BFS pathfinding
  - `ExplorationModePOMDP` class - frontier-based exploration:
    - `update()` - records transitions, selects actions
    - `_select_action()` - frontier or backtrack logic
    - `record_movement_result()` - wall detection
  - `ExplorationResult` dataclass with graph stats
- [x] Mode switching preserved for compatibility
- [x] Legacy code moved to `legacy/exploration_mode_old.py`

**Key insight:** Topological frontier exploration is the standard approach in robotics for a reason - it's simple, robust, and complete. No need for complex depth-based door detection.

### Phase 9: Observation Encoding Improvements

**Problem Discovered:** YOLO object detection histograms are too noisy for stable location signatures. Testing showed 32 locations created while rotating in a single room. Issues:
- Objects appear/disappear at frame edges
- Confidence scores fluctuate frame-to-frame
- Different angles show different object subsets
- Empty walls/floors create spurious "unique" signatures

#### Option A: Semantic Image Embeddings (TESTED - INADEQUATE)

- [x] `pomdp/image_encoder.py`:
  - `ImageEncoder` class using CLIP ViT-B/32 (512-dim)
  - `DINOv2Encoder` class using DINOv2-small (384-dim)
  - `encode_frame(frame) -> np.ndarray` - extract embedding vector
  - Cosine similarity on embeddings for location matching

**Testing Results (3D Simulator):**

| Model | Living↔Kitchen | Living↔Bathroom | Kitchen↔Bedroom | Threshold |
|-------|----------------|-----------------|-----------------|-----------|
| CLIP | 0.931 | 0.916 | 0.991 | 0.85 |
| DINOv2 | 0.817 | 0.880 | 0.807 | 0.85 |

**Key Finding:** Semantic embeddings fail for place recognition because:
- CLIP/DINOv2 trained for "what is this image?" (semantic)
- Navigation needs "have I been here before?" (spatial)
- Indoor rooms dominated by walls, floors, partial furniture
- All rooms look like "indoor space" to semantic models
- Viewpoint changes drastically affect embeddings

**Conclusion:** Semantic embeddings are not the right tool for embodied navigation.

#### Option B: ORB Keyframe Place Recognition (TESTED - PROMISING)

- [x] `pomdp/place_recognizer.py`:
  - `ORBPlaceRecognizer` class - local feature matching
  - `preload_room(frames, name)` - bootstrap with known rooms
  - `recognize(frame)` - match against keyframe database
  - Lowe's ratio test for robust matching

**Testing Results (3D Simulator):**
- 50% accuracy on synthetic rooms
- Works well on rooms with texture (Hallway: 500 features)
- Fails on plain rooms (Bedroom: 19 features, Bathroom: 18 features)
- **Key insight:** ORB needs real-world texture (edges, corners, patterns)

**Why ORB is fundamentally right:**
- Local features are viewpoint-robust
- Captures corners, edges, texture (not semantics)
- "Have I seen this pattern before?" vs "What room is this?"
- Industry standard for visual place recognition and SLAM

**Limitation:** Our synthetic 3D simulator lacks real-world texture complexity.

#### Option C: Full Place Recognition Stack (RECOMMENDED)

Based on robotics literature and testing insights:

1. **ORB/AKAZE Keyframes** - viewpoint-stable local features
   - Keyframe graph: nodes = keyframes, edges = visual similarity + motion
   - Works well in real environments with natural texture

2. **Object-Centric Spatial Layouts** - structured YOLO
   - Not just "what objects" but "where objects are relative to you"
   - `{couch:left:near, tv:center:far}` as location signature

3. **Motion-Conditioned Recognition** - sequences, not frames
   - Embed N-frame windows (0.5-1.0s)
   - CSCG explicitly models (action, observation) sequences
   - Drastically reduces perceptual aliasing

4. **Visual Odometry** - rough is enough
   - "I moved forward", "I rotated left"
   - Stabilizes place matching
   - Improves CSCG transition learning

**Why this stack works:**
```
ORB features + keyframes → place IDs
YOLO object layouts     → semantic hints
Optical flow / VO       → motion cues
CSCG                    → resolves aliasing + hierarchy
VBGS                    → local geometry refinement
JAX POMDP              → action selection
```

This is how biological and robotic navigation actually works.

### Phase 9b: 3D Simulator Development

- [x] `simulator/simple_3d.py`:
  - Raycasting 3D renderer with textured walls
  - 5-room house layout (Living Room, Kitchen, Hallway, Bedroom, Bathroom)
  - Furniture rendering (sofa, bed, fridge, bathtub, etc.)
  - Collision detection for walls and furniture
  - Minimap overlay

- [x] `simulator/glb_simulator.py`:
  - **NEW: GLB model-based 3D renderer using pyrender**
  - Loads realistic house model (simple_house.glb)
  - First-person navigation with WASD controls
  - Real textures, furniture, lighting
  - Camera positioning and orientation control

- [x] `test_full_pipeline.py`:
  - Autonomous exploration test with CSCG
  - Visual feedback with OpenCV
  - Mode switching (exploration/hunting)
  - Debug output for place recognition

**ORB Testing Results Comparison:**

| Environment | Avg Features | Recognition Accuracy | Loop Closure |
|-------------|--------------|---------------------|--------------|
| Simple Raycaster | 20-500 | 50% | Unreliable |
| **GLB House Model** | **463.7** | **87.5%** | **100% confidence** |

**Key Finding:** Realistic 3D models provide the texture ORB needs.

**GLB Simulator Advantages:**
- Consistent 500 features per frame (vs variable 20-500)
- Proper lighting and shadows create natural gradients
- Realistic furniture with textures
- Successfully detects loop closure (return to start)

**Simple Raycaster Limitations:**
- Solid-color walls lack feature points
- Simple textures don't generate enough keypoints
- Furniture rendered as flat colored boxes

---

## Phase 10: CSCG + VBGS Unified Cognitive Mapping (IN PROGRESS)

### Overview

Integrate **Clone-Structured Cognitive Graphs (CSCG)** and **Variational Bayesian Gaussian Splatting (VBGS)** together for robust place recognition and topological mapping.

**Key Insight:** Use VBGS as "local place models" without requiring accurate metric poses (markerless Option 1). CSCG handles the graph structure and perceptual aliasing.

### Architecture

```
Frame → CLIP embedding → Token clustering → CSCG inference → Clone state belief
              ↓                                      ↓
         Keyframe?                              Graph structure
              ↓                                      ↓
    VBGS local place model              POMDP belief update
              ↓                                      ↓
    - ELBO (place evidence)              - Location belief
    - Sub-area structure                 - Action selection
```

**Two Complementary Systems:**

| Component | Role | Output |
|-----------|------|--------|
| **CSCG** | Graph structure + clone disambiguation | T matrix (transitions), E matrix (emissions), clone states |
| **VBGS** | Local place appearance model | ELBO (fit quality), rendered expected view, sub-area Gaussians |

### Phase 10a: CSCG Implementation ✓ IN PROGRESS

**Reference:** [vicariousinc/naturecomm_cscg](https://github.com/vicariousinc/naturecomm_cscg)

**Core Concept:** Clone-structured Hidden Markov Model (CHMM)
- Multiple "clone" states can emit the same observation
- Resolves perceptual aliasing (two hallways look the same)
- Learns transition structure T(z'|z,a) and emission structure E(x|z)

**Files to Create:**
- [ ] `mapping/__init__.py` - Mapping package
- [ ] `mapping/cscg/__init__.py` - CSCG package
- [ ] `mapping/cscg/chmm.py` - Core CHMM implementation:
  - `CHMM` class (adapted from naturecomm_cscg):
    - `__init__(n_clones, n_obs, n_actions)` - initialize with clone counts per observation
    - `forward(x, a)` - forward message passing, returns log-likelihood
    - `backward(x, a)` - backward message passing
    - `decode(x, a)` - MAP state sequence via Viterbi
    - `learn_em_T(x, a)` - EM learning for transition matrix
    - `learn_em_E(x, a)` - EM learning for emission matrix
    - `sample(length)` - generate sequences from model
    - `bridge(state1, state2)` - path planning between states
  - Numba JIT optimization for real-time performance
- [ ] `mapping/cscg/tokenizer.py` - Observation tokenization:
  - `EmbeddingTokenizer` class:
    - `__init__(n_tokens, embedding_dim)` - k-means clustering setup
    - `fit(embeddings)` - learn token clusters from CLIP embeddings
    - `tokenize(embedding)` - map embedding to discrete token
    - `add_observation(embedding)` - online cluster update
  - Combines CLIP embedding + YOLO histogram for richer tokens
- [ ] `mapping/cscg/cscg_world_model.py` - Integration with POMDP:
  - `CSCGWorldModel` class:
    - Wraps CHMM with POMDP interface
    - `localize(obs, action, frame)` - returns LocalizationResult
    - `get_clone_belief()` - soft belief over clone states
    - `get_community_belief()` - room-level belief (graph modularity)
    - Clone states → POMDP locations
    - T matrix → B matrix (transitions)
    - E matrix → A matrix (observations)

**Integration Points:**
- Replace `TopologicalMap` location matching with CSCG clone inference
- CSCG T matrix provides learned transition dynamics
- Clone states disambiguate perceptually similar locations

### Phase 10b: VBGS Local Place Models

**Reference:** `vbgs/` folder (already in project)

**Core Concept:** Each "place" (observation token cluster) gets its own VBGS model
- No global poses required - just local image consistency
- VBGS ELBO indicates how well current frame fits place model
- Gaussian components = sub-areas within place

**Files to Create:**
- [ ] `mapping/vbgs_place/__init__.py` - VBGS place model package
- [ ] `mapping/vbgs_place/place_model.py`:
  - `VBGSPlaceModel` class:
    - `__init__(n_components)` - initialize Gaussian mixture
    - `update(frame)` - continual learning from keyframe
    - `compute_elbo(frame)` - how well does frame fit this place?
    - `get_expected_view(shape)` - render expected appearance
    - `get_subarea_assignments()` - which Gaussian component dominates?
  - Uses `vbgs.model.train.fit_delta_gmm_step` for online updates
  - Accumulates sufficient statistics across keyframes
- [ ] `mapping/vbgs_place/keyframe_selector.py`:
  - `KeyframeSelector` class:
    - `should_add_keyframe(frame, embedding)` - viewpoint change detection
    - Uses embedding distance + frame difference
    - Rate limiting (max N keyframes per second)
- [ ] `mapping/vbgs_place/place_manager.py`:
  - `PlaceManager` class:
    - `places: Dict[int, VBGSPlaceModel]` - one VBGS per token/clone
    - `update_place(place_id, frame)` - add keyframe to place model
    - `compute_place_evidence(frame)` - ELBO for all places
    - `get_best_place(frame)` - argmax ELBO

**Integration Points:**
- VBGS ELBO provides additional evidence for CSCG inference
- Sub-area Gaussians enable finer localization within places
- Expected view rendering for debugging/visualization

### Phase 10c: Unified Integration

**Files to Modify:**
- [ ] `pomdp/world_model.py` - Add CSCG+VBGS backend option:
  - `WorldModel.__init__(backend='topological'|'cscg')` - backend selection
  - CSCG backend uses `CSCGWorldModel` internally
  - VBGS evidence fused into localization likelihood
- [ ] `person_hunter_pomdp.py` - Use unified system:
  - `POMDPController` uses CSCG world model
  - Visualization shows clone states + VBGS evidence
  - Keyframe collection during exploration

**Unified Localization Flow:**
```python
def localize(self, obs, action, frame):
    # 1. Extract CLIP embedding
    embedding = self.image_encoder.encode(frame)

    # 2. Tokenize to discrete observation
    token = self.tokenizer.tokenize(embedding)

    # 3. CSCG clone state inference
    clone_belief = self.chmm.forward(token, action)

    # 4. VBGS place evidence (optional, for high-confidence frames)
    if self.use_vbgs:
        vbgs_evidence = self.place_manager.compute_place_evidence(frame)
        clone_belief = self.fuse_evidence(clone_belief, vbgs_evidence)

    # 5. Update keyframe if viewpoint changed
    if self.keyframe_selector.should_add_keyframe(frame, embedding):
        place_id = clone_belief.argmax()
        self.place_manager.update_place(place_id, frame)

    # 6. Return POMDP-compatible result
    return LocalizationResult(
        belief=clone_belief,
        location_id=clone_belief.argmax(),
        vfe=self.compute_vfe(clone_belief, token),
        ...
    )
```

### Phase 10d: Testing & Validation

- [ ] Test CSCG learning on recorded exploration sequences
- [ ] Test VBGS place models with stationary drone views
- [ ] Test integrated system with ground test mode
- [ ] Compare localization accuracy: TopologicalMap vs CSCG+VBGS
- [ ] Verify clone disambiguation in similar-looking areas

### Key Benefits

| Feature | TopologicalMap (current) | CSCG+VBGS (new) |
|---------|-------------------------|-----------------|
| Perceptual aliasing | ✗ Fails | ✓ Clone states disambiguate |
| Graph structure | Manual edges | ✓ Learned from experience |
| Place appearance | CLIP embedding only | ✓ CLIP + VBGS local model |
| Sub-area localization | ✗ None | ✓ VBGS Gaussian components |
| VFE computation | Sparse A matrix | ✓ VBGS ELBO (dense) |
| Room hierarchy | ✗ None | ✓ Graph community detection |

---

## Next Steps (Priority Order)

### Immediate: Real-World Testing
1. **Test ORB place recognition with real video/webcam**
   - Real environments have natural texture that ORB excels at
   - Record exploration video, test place recognition offline
   - Compare accuracy vs synthetic simulator

2. **Integrate motion cues**
   - Track optical flow between frames
   - Use action history to condition place recognition
   - CSCG benefits from (action, observation) sequences

### Short-term: Enhanced Place Recognition

3. **Object-centric spatial layouts**
   - Enhance YOLO encoding: `{object: position, distance}`
   - Spatial layout more stable than raw detection histograms

4. **Visual odometry integration**
   - Even rough VO helps: "moved forward", "rotated left"
   - Stabilizes place matching during motion

### Medium-term: Full Stack Integration

5. **CSCG with real observations**
   - Clone-structured HMM resolves perceptual aliasing
   - Learns transition structure from exploration

6. **VBGS local place models**
   - Each keyframe cluster gets local Gaussian model
   - ELBO provides place evidence

---

## Future Phases

### Phase 11: Semantic SLAM with Active Inference (IN PROGRESS)

**Goal:** Prior-driven exploration that finds all expected rooms and objects.

**Architecture (Layered):**
```
┌─────────────────────────────────────────────────────────┐
│  SEMANTIC LAYER (TOP) - mapping/semantic_world_model.py │
│  • Room type priors & classification                    │
│  • Object tracking per place                            │
│  • EFE pragmatic term: KL[Q(rooms)||P(rooms)]           │
│  • "Find kitchen, bedroom, bathroom..."                 │
├─────────────────────────────────────────────────────────┤
│  CSCG LAYER (MIDDLE) - mapping/cscg/cscg_world_model.py │
│  • ORB place recognition (keyframes)                    │
│  • Clone state disambiguation                           │
│  • Transition learning (T matrix)                       │
│  • EFE epistemic term: reduce transition uncertainty    │
│  • "Where am I? What connects to what?"                 │
├─────────────────────────────────────────────────────────┤
│  PERCEPTION LAYER (BOTTOM)                              │
│  • ORB keypoint extraction                              │
│  • YOLO object detection                                │
│  • Frame preprocessing                                  │
│  • "What do I see?"                                     │
└─────────────────────────────────────────────────────────┘
```

**Data Flow:**
```
Frame → ORB/YOLO → CSCG place ID → Semantic (room type + objects)
                                          ↓
                           EFE = epistemic(CSCG) + pragmatic(priors)
                                          ↓
                                    Action selection
```

**Core Insight:** Frame exploration as Expected Free Energy minimization:

```
EFE(action) = w_e * G_epistemic + w_p * G_pragmatic

G_epistemic = entropy of predicted transitions (from CSCG)
            = "Explore unknown transitions"

G_pragmatic = KL[Q(rooms) || P(rooms)] + KL[Q(objects|room) || P(objects|room)]
            = "Find all expected rooms and objects"
```

**Semantic Priors:**
```python
prior = SemanticPrior(
    room_types=["living_room", "kitchen", "bedroom", "bathroom"],
    objects_per_room={
        "kitchen": ["refrigerator", "oven", "sink", "table"],
        "bedroom": ["bed", "chair", "tv"],
        "living_room": ["couch", "tv", "chair"],
        "bathroom": ["toilet", "sink"],
    },
    reference_images={  # For ORB-based room classification
        "kitchen": ["refs/kitchen_1.jpg", "refs/kitchen_2.jpg"],
        "bedroom": ["refs/bedroom_1.jpg"],
        ...
    }
)
```

**Implementation:**

- [x] `mapping/semantic_world_model.py`:
  - `SemanticWorldModel` class - wraps CSCG, adds semantic layer
  - `SemanticPrior` dataclass - expected rooms and objects
  - `SemanticPlace` dataclass - room type, objects per CSCG place
  - `ObjectInstance` dataclass - tracked objects with positions
  - `update(frame, action, yolo_detections)` - calls CSCG + semantic update
  - `compute_efe()` - combines CSCG epistemic + semantic pragmatic
  - `get_exploration_action()` - minimize combined EFE
  - `get_exploration_urgency()` - based on semantic coverage
  - Room type classification via ORB matching against references
  - YOLO object tracking per CSCG place

**Unified EFE Computation:**
```python
def compute_efe(self, action):
    # Get epistemic from CSCG (transition uncertainty)
    cscg_action, cscg_probs = self.cscg.get_exploration_target()
    epistemic = -np.log(cscg_probs[action] + 1e-10)  # Lower prob = more info gain

    # Compute pragmatic from semantic priors
    pragmatic = 0.0
    for predicted_place in self.cscg.predict_destinations(action):
        sem_place = self.places.get(predicted_place)
        if sem_place is None or sem_place.room_type is None:
            pragmatic -= 0.5  # Unknown room type
        elif sem_place.room_type not in self._found_room_types():
            pragmatic -= 1.0  # New room type!
        pragmatic -= 0.3 * len(self._missing_objects(sem_place))

    return (1 - self.pragmatic_weight) * epistemic + self.pragmatic_weight * pragmatic
```

**Exploration Flow:**
```
1. Initialize: CSCG for places, Semantic layer with priors
2. Each step:
   a. CSCG localizes to place ID (ORB matching)
   b. Semantic layer classifies room type (ORB vs reference images)
   c. YOLO updates object list for current place
   d. Compute EFE = CSCG epistemic + Semantic pragmatic
   e. Select action minimizing EFE
3. Stop when: all expected rooms found AND key objects located
```

**Files:**
- [x] `mapping/semantic_world_model.py` - Semantic layer wrapping CSCG
- [x] `mapping/cscg/cscg_world_model.py` - CSCG layer (already exists)
- [ ] `refs/` directory - Reference images for room types
- [x] Integration in `test_full_pipeline.py` - `--semantic` flag

**Completed Implementation (Dec 2025):**

- [x] **Layered architecture**: Semantic layer wraps CSCG (not separate)
- [x] **Object-based room inference**: YOLO detects objects → infer room type
  ```python
  OBJECT_TO_ROOM = {
      "toilet": "bathroom",
      "bed": "bedroom",
      "couch": "living_room",
      "refrigerator": "kitchen",
      ...
  }
  ```
- [x] **YOLO integration in test pipeline**: Runs YOLOv8n on each frame
- [x] **Room type voting**: Each CSCG place accumulates object→room votes
- [x] **Stuck-at-boundary fix**: Detects repeated blocks, forces rotation
- [x] **Unified EFE**: `(1-w)*epistemic + w*pragmatic` with configurable weight

**Room Classification Methods (prioritized):**
1. **Object inference** (implemented) - toilet→bathroom, bed→bedroom
2. **Reference images** (optional) - ORB match against room photos
3. **Simulator hint** (testing only) - ground truth from GLB position

### Phase 11b: Spatial Object Mapping (PLANNED)

**Goal:** Map objects to 3D world coordinates, not just "object X is at place Y".

**Current State:**
```
YOLO detects "bed" at pixel (320, 400)
  → Store: Place_5 has "bed" at normalized frame position (0.5, 0.83)
  → Infer: Place_5 is "bedroom"
```

**Target State:**
```
YOLO detects "bed" at pixel (320, 400)
  → Estimate depth: ~3 meters (from monocular depth or bbox size)
  → Drone pose: (x=100, z=200, yaw=45°)
  → Project to world: bed at absolute coords (102.1, 202.1)
  → Store: "bed" at world position (102.1, 202.1)
```

**Implementation Options:**

#### Option A: Monocular Depth Estimation
- Use MiDaS or DepthAnything to estimate depth from single image
- More accurate but computationally expensive
- Requires GPU for real-time performance

```python
# Pseudo-code
depth_map = midas(frame)  # H x W depth estimates
bbox_depth = depth_map[cy, cx]  # Depth at object center
world_pos = project_to_world(cx, cy, bbox_depth, drone_pose)
```

#### Option B: Bounding Box Size Heuristics
- Estimate distance from known object sizes
- Fast, no extra model needed
- Less accurate but sufficient for coarse mapping

```python
# Known object heights (meters)
OBJECT_HEIGHTS = {
    "person": 1.7,
    "chair": 0.9,
    "refrigerator": 1.8,
    "bed": 0.5,  # height when lying flat
    "toilet": 0.4,
    ...
}

def estimate_distance(class_name, bbox_height_pixels, frame_height, fov_y):
    real_height = OBJECT_HEIGHTS.get(class_name, 0.5)
    # Pinhole camera model: distance = (real_height * focal_length) / pixel_height
    focal_length = frame_height / (2 * tan(fov_y / 2))
    distance = (real_height * focal_length) / bbox_height_pixels
    return distance
```

#### Option C: Hybrid Approach (Recommended)
- Use bbox heuristics for coarse estimates (always available)
- Refine with monocular depth when GPU available
- Triangulate from multiple viewpoints for accuracy

**Data Structures:**

```python
@dataclass
class SpatialObject:
    class_name: str
    world_position: Tuple[float, float]  # (x, z) in world coords
    position_confidence: float  # Higher with more observations
    observation_count: int
    last_seen_place_id: int

    # For triangulation
    observations: List[ObjectObservation]  # Multiple viewpoints

@dataclass
class ObjectObservation:
    place_id: int
    drone_pose: Tuple[float, float, float]  # (x, z, yaw)
    pixel_position: Tuple[float, float]  # Normalized (cx, cy)
    estimated_distance: float
    confidence: float
```

**Triangulation from Multiple Views:**
```
View 1: drone at (0, 0, 0°), object at pixel (0.7, 0.5) → ray at 20° right
View 2: drone at (2, 0, 45°), object at pixel (0.3, 0.5) → ray at -25° left
  → Intersect rays → object at world (1.5, 3.2)
```

**Files to Create:**
- [ ] `mapping/spatial_objects.py` - SpatialObject tracking
- [ ] `mapping/depth_estimation.py` - Monocular depth (MiDaS/DepthAnything)
- [ ] `mapping/object_triangulation.py` - Multi-view object localization

**Integration with Semantic Layer:**
```python
class SemanticWorldModel:
    def update(self, frame, action, yolo_detections, drone_pose):
        # ... existing CSCG + room inference ...

        # NEW: Spatial object mapping
        if yolo_detections and drone_pose:
            for det in yolo_detections:
                distance = self.estimate_distance(det)
                world_pos = self.project_to_world(det, distance, drone_pose)
                self.spatial_objects.update(det.class_name, world_pos, drone_pose)
```

### Phase 12: Visual Odometry

- [ ] Add lightweight VO (optical flow + essential matrix)
- [ ] Estimate relative pose between frames
- [ ] Feed motion cues into CSCG transition learning
- [ ] No global SLAM needed - local consistency sufficient

### Phase 13: Room Hierarchy via Community Detection

- [ ] Run graph community detection on learned CSCG structure
- [ ] Communities = room-level abstractions
- [ ] High-level POMDP for room-to-room navigation

### Phase 14: Full Hierarchical Controller

```
High-level (room policy):
  - "Go to kitchen to find human"
  - Uses room community beliefs

Mid-level (place navigation):
  - "Navigate through clone states to reach kitchen"
  - Uses CSCG graph for path planning (bridge())

Low-level (motor control):
  - "Execute approach action"
  - Uses InteractionMode POMDP
```

---

## Dependencies

Add to environment.yml:
```yaml
- pip:
    - jax[cpu]>=0.4.20    # or jax[cuda12] for GPU
    - jaxlib>=0.4.20
```

For CSCG (Phase 10):
```yaml
- pip:
    - networkx            # For community detection
```

For VBGS (Phase 12) - **separate environment**:
```yaml
# Create separate conda env: conda create -n vbgs python=3.10
- pip:
    - jax[cuda12]         # VBGS requires JAX with CUDA
    - torch               # For rendering components
```

---

## Files to Create

| File | Purpose | Phase |
|------|---------|-------|
| `pomdp/__init__.py` | Package init | 1 ✓ |
| `pomdp/config.py` | Constants, thresholds | 1 ✓ |
| `pomdp/core.py` | JAX JIT belief functions | 1 ✓ |
| `pomdp/observation_encoder.py` | YOLO → fixed tokens | 1 ✓ |
| `pomdp/topological_map.py` | TopologicalMap class | 2 ✓ |
| `pomdp/similarity.py` | Observation similarity | 2 ✓ |
| `pomdp/map_persistence.py` | Save/load maps | 2 ✓ |
| `pomdp/world_model.py` | Location belief POMDP | 3 ✓ |
| `pomdp/human_search.py` | Human location POMDP | 4 ✓ |
| `pomdp/interaction_mode.py` | Action selection POMDP | 5 ✓ |
| `safety/__init__.py` | Package init | 6 ✓ |
| `safety/overrides.py` | Safety checks | 6 ✓ |
| `utils/__init__.py` | Package init | 7 ✓ |
| `utils/frame_grabber.py` | Threaded frame grabber | 7 ✓ |
| `person_hunter_pomdp.py` | Main integration script | 7 ✓ |
| `pomdp/exploration_mode.py` | Exploration mode POMDP | 8 ✓ |
| `mapping/cscg/` | CSCG cognitive maps | 10 |
| `pose/` | Pose estimation | 11 |
| `mapping/vbgs/` | VBGS integration | 12 |

## Files to Modify

| File | Change |
|------|--------|
| `environment.yml` | Add JAX dependencies |

## Reference Files (unchanged)

| File | Why |
|------|-----|
| `person_hunter_safe.py` | Reference for main loop, YOLO integration, RC control |

---

## Critical Bug Fixes (Dec 2025 - Jan 2026)

All fixes implemented and tested.

### Issue 1: Occupancy map uint8 overflow ✓ FIXED
**File:** `utils/occupancy_map.py`

**Problem:** Grid stored as `dtype=np.uint8`, arithmetic wraps around before `min/max` clamp.
```python
current = self.grid[y, x]  # uint8
new_val = min(255, current + free_increment)  # WRAPS before min!
```

**Symptoms:**
- Free cells suddenly look occupied (or vice versa)
- Frontier/exploration heuristics unstable
- "Blocked everywhere" / oscillation common

**Fix:** Convert to int before arithmetic:
```python
current = int(self.grid[y, x])
new_val = min(255, current + self.config.free_increment)
self.grid[y, x] = np.uint8(new_val)
```

### Issue 7: Doorways painted as obstacles ✓ FIXED (Jan 2026)
**Files:** `utils/occupancy_map.py`, `pomdp/frontier_explorer.py`

**Problem:** Depth sensing at an angle would mark doorway openings as obstacles. When looking at a doorway from the side, rays hit walls on either side and the 3x3 obstacle dilation "painted over" the actual opening.

**Symptoms:**
- Doorway band diagnostic shows `[    #####  ]` (5 consecutive obstacles)
- A* cannot find path through doorways
- Drone gets stuck at room boundaries

**Root Cause:**
- Depth-based obstacles used same 3x3 dilation as collision obstacles
- No distinction between "I see something far away" vs "I hit a wall"
- Single depth hit could forbid entire doorway

**Fix (Two-Tier Obstacle System):**

1. **Depth-sensed obstacles** (low confidence):
   - Single cell marking (no dilation)
   - Floor at value 60 (not 0)
   - Can be overwritten by free-space carving
   ```python
   new_val = max(self.config.depth_obstacle, current - self.config.depth_occupied_increment)
   ```

2. **Collision-confirmed obstacles** (high confidence):
   - 3x3 dilation (unchanged)
   - Value 0, locked with high visit count
   - Cannot be overwritten by depth

3. **Grid obstacle gating** (frontier explorer):
   - Changed threshold from `< 50` to `< 30`
   - Only blocks on collision-confirmed obstacles
   ```python
   # Only block on COLLISION-CONFIRMED obstacles (value < 30)
   # Depth-sensed obstacles floor at ~60, so this avoids false blocking
   if cell_val < 30:
   ```

**Results:**
- Doorway bands now show 0 obstacle cells (vs 5 before)
- Total obstacles reduced from 51 to 8 (collision-confirmed only)
- Room transitions now pathable

### Issue 8: Escape scan mutating simulator state ✓ FIXED (Jan 2026)
**Files:** `simulator/glb_simulator.py`, `test_full_pipeline.py`

**Problem:** When scanning 360° to find escape direction, each test move permanently changed simulator state (room transitions, position drift).

**Fix:** Added snapshot/restore methods:
```python
def get_state_snapshot(self) -> dict:
    return {'x': self.x, 'y': self.y, 'z': self.z, 'yaw': self.yaw, ...}

def restore_state_snapshot(self, snapshot: dict):
    self.x = snapshot['x']
    ...
```

### Issue 9: Escape mode counting blocked steps ✓ FIXED (Jan 2026)
**File:** `pomdp/frontier_explorer.py`

**Problem:** Escape step counter decremented even when movement was blocked, so escape would "complete" while still stuck.

**Fix:** `report_escape_move_result()` only decrements on successful forward:
```python
if action == 1:  # FORWARD
    if moved:
        self._escape_steps_remaining -= 1
    else:
        # Blocked during escape - abort and re-scan
        self._escape_mode = False
        self._needs_escape_scan = True
```

### Issue 11: Frontier Explorer Tuning ✓ FIXED (Jan 2026)
**Files:** `pomdp/frontier_explorer.py`

**Problem:** Default escape thresholds and frontier scoring needed tuning for better room exploration.

**Changes:**

1. **Escape frequency reduced:**
   - Short-term turns: 1-5 (was 1-4)
   - Medium-term scan: 6-11 (was 5-9)
   - Long-term clear: 12+ (was 10+)

2. **Doorway detection and bonus:**
   - `_is_doorway_cluster()` detects elongated frontiers (aspect ratio > 1.5)
   - 0.4m distance bonus for doorway clusters in scoring
   - Prioritizes room transitions over open space exploration

3. **Room transition tracking:**
   - `RoomTransitionTracker` class tracks door crossings as first-class events
   - `report_room(room_name, position)` API for pipeline integration
   - Stats: total transitions, rooms visited, unique doorways used

4. **Center-of-gap bias:**
   - `find_gap_center()` scans perpendicular to heading
   - `get_gap_steering_correction()` returns heading adjustment (max ±8°)
   - Only applies for narrow gaps (< 0.6m width)

**Test Results (150 frames):**
- 2 rooms visited (Kitchen, Living Room)
- 2 door crossings
- 17.7% doorway cluster detection rate
- All recovery mechanisms working

### Issue 10: Oscillation after escape ✓ FIXED (Jan 2026)
**File:** `pomdp/frontier_explorer.py`

**Problem:** After successful escape, A* would immediately re-plan to same blocked target, causing oscillation.

**Root Cause:** Target was blacklisted, but the *approach direction* wasn't remembered. New path would use same blocked approach.

**Fix (Blocked-Edge Memory):**
```python
# Record blocked edges with TTL
self._blocked_edges: Dict[Tuple[Tuple[int,int], Tuple[int,int]], int] = {}

def record_blocked_edge(self, from_cell, to_cell):
    edge = (from_cell, to_cell)
    self._blocked_edges[edge] = self._frame_count + self._blocked_edge_ttl

# A* skips blocked edges
for nb in self._neighbors(grid, current):
    if nb in self._get_blocked_neighbors(current):
        continue
```

Also: `report_escape_move_result()` clears path but keeps target, forcing fresh A* from new position.

### Issue 2: Ellipsis in escape_actions list ✓ CHECKED
**File:** `pomdp/exploration_mode.py`

**Status:** Checked - list is correct in current version (no Ellipsis bug found).

### Issue 3: Depth normalization is per-frame (unstable collision detection) ✓ FIXED
**File:** `utils/depth_estimator.py`

**Problem:** Per-frame min/max normalization makes "blocked" threshold non-stationary.

**Fixes Applied:**
- Use percentiles (2nd/98th) instead of min/max for robust normalization
- Temporal EMA smoothing of collision scores
- Debounce "blocked" for K consecutive frames (default 3)

### Issue 4: Room classification can never predict "hallway" ✓ FIXED
**File:** `mapping/semantic_world_model.py`

**Fixes Applied:**
- Added hallway detection via negative evidence (few objects + many visits)
- Added object confidence decay (objects not seen recently lose confidence)
- Objects must be observed at least 2 times to count for room voting
- Objects with confidence < 0.1 are removed

### Issue 5: ORB keyframe creation lacks spatial gating ✓ FIXED
**File:** `pomdp/place_recognizer.py`

**Fixes Applied:**
- Added cumulative yaw tracking since last keyframe
- Require minimum yaw change (~17°) OR M consecutive low-match frames (default 5)
- Reset yaw accumulator when creating new keyframe
- Added `yaw_delta` parameter to `recognize()` method

### Issue 6: VBGS not loading
**Problem:** Missing `jaxtyping` dependency.

**Fix:** `pip install jaxtyping`

---

## Collision Avoidance Module (Dec 2025)

**File:** `utils/collision_avoidance.py`

Implements hybrid collision avoidance combining:
1. **Optical Flow TTC** - Fast reflex for approaching surfaces
2. **Monocular Depth Gate** - Confirmatory gate using depth percentiles
3. **Yaw Scan Escape** - Find open direction when stuck

### Key Classes

| Class | Purpose |
|-------|---------|
| `OpticalFlowTTC` | Computes time-to-contact from optical flow expansion |
| `CollisionAvoidance` | Hybrid risk assessment + escape state machine |
| `RiskLevel` | Enum: NONE, LOW, MEDIUM, HIGH, CRITICAL |

### Usage

```python
from utils.collision_avoidance import create_collision_reflex

# Create with depth estimator for hybrid mode
collision = create_collision_reflex(depth_estimator=depth_estimator)

# Assess risk from current frame
state = collision.assess_risk(frame, depth_map)

# Get safe action (may override desired action)
safe_action, reason = collision.get_safe_action(desired_action, state)
```

### Collision State Machine

```
if risk >= CRITICAL:
    emergency_backoff
elif risk >= HIGH:
    block forward, yaw_scan to find open direction
elif risk >= MEDIUM:
    pause if sustained, else proceed cautiously
else:
    proceed normally
```

### Why Hybrid Works

- **Optical Flow** catches "approaching a surface" even if depth fails (glossy walls)
- **Monocular Depth** catches "static near obstacles" even if flow is low
- Combined = much more robust than either alone

---

## Phase 15: End-to-End Hunting Pipeline (FIRM PLAN)

### Current State (Jan 2026)

**Working:**
- Frontier-based exploration with A* pathfinding
- Two-tier occupancy mapping (depth vs collision)
- Room transition tracking and door crossing events
- Escape strategies with blocked-edge memory
- GLB simulator with realistic 3D house model

**Partially Working:**
- CSCG place recognition (ORB-based)
- Semantic room classification (YOLO object inference)
- Human search POMDP (belief tracking)
- Interaction mode POMDP (action selection)

**Not Yet Integrated:**
- Exploration → Hunting mode switch
- Person/cat detection triggering hunt
- Approach and interaction behaviors
- Return-to-patrol after interaction

### Target: Autonomous Person/Cat Hunting

```
┌─────────────────────────────────────────────────────────────────┐
│                    TELLO HUNT PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 1: EXPLORATION                                           │
│  ├─ Frontier explorer builds occupancy map                      │
│  ├─ Room transitions tracked (door crossings)                   │
│  └─ Continue until: target detected OR coverage threshold       │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 2: TARGET ACQUISITION                                    │
│  ├─ YOLO detects person/cat in frame                           │
│  ├─ Human search POMDP updates belief: "target at location X"  │
│  └─ Trigger: switch to HUNTING mode                            │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 3: APPROACH                                              │
│  ├─ Interaction POMDP selects: APPROACH action                 │
│  ├─ Visual servoing: center target in frame                    │
│  ├─ Distance control: approach to ~2m                          │
│  └─ Collision avoidance active throughout                      │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 4: INTERACTION                                           │
│  ├─ At target distance: switch to INTERACT state               │
│  ├─ Execute interaction: wiggle, LED pattern, hover            │
│  ├─ Monitor target response (movement, attention)              │
│  └─ Backoff if target leaves or gets too close                 │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 5: RETURN TO PATROL                                      │
│  ├─ After interaction timeout OR target lost                   │
│  ├─ Resume exploration from current position                   │
│  └─ Update human belief: decay confidence at last location     │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Steps

#### Step 1: Mode Switching Logic (Priority: HIGH)

**File:** `test_full_pipeline.py` (modify existing)

Add mode switching based on YOLO detection:

```python
class PipelineController:
    def __init__(self):
        self.mode = "EXPLORATION"  # or "HUNTING"
        self.target_class = "person"  # or "cat"
        self.target_lost_frames = 0
        self.interaction_timer = 0

    def update(self, frame, yolo_results):
        # Check for target
        target_detected = self._detect_target(yolo_results)

        if self.mode == "EXPLORATION":
            if target_detected:
                self.mode = "HUNTING"
                print(f"[MODE] EXPLORATION -> HUNTING ({self.target_class} detected)")
            else:
                return self.frontier_explorer.choose_action(...)

        elif self.mode == "HUNTING":
            if not target_detected:
                self.target_lost_frames += 1
                if self.target_lost_frames > 30:  # ~1 second
                    self.mode = "EXPLORATION"
                    print("[MODE] HUNTING -> EXPLORATION (target lost)")
            else:
                self.target_lost_frames = 0
                return self.interaction_pomdp.update(...)
```

#### Step 2: Visual Servoing for Approach (Priority: HIGH)

**File:** `pomdp/visual_servoing.py` (new)

Simple proportional control to center target and approach:

```python
@dataclass
class ServoingCommand:
    yaw: int      # Turn to center target horizontally
    fb: int       # Forward/back to reach target distance
    ud: int       # Up/down to center target vertically

def compute_servoing(bbox, frame_shape, target_distance_m=2.0):
    """Compute RC commands to approach and center target."""
    cx, cy, w, h = bbox  # Normalized 0-1

    # Yaw: center target horizontally
    x_error = cx - 0.5
    yaw = int(-x_error * 60)  # P-gain, max ±30

    # Forward/back: approach to target distance
    # Estimate distance from bbox height (larger = closer)
    estimated_dist = estimate_distance_from_bbox(h)
    dist_error = estimated_dist - target_distance_m
    fb = int(dist_error * 20)  # P-gain, max ±40
    fb = max(-40, min(40, fb))

    # Vertical: center target (optional, keep level for stability)
    y_error = cy - 0.5
    ud = int(-y_error * 20)

    return ServoingCommand(yaw=yaw, fb=fb, ud=ud)
```

#### Step 3: Integrate Interaction POMDP (Priority: MEDIUM)

**File:** `pomdp/interaction_mode.py` (already exists)

Current implementation has the right structure. Need to:

1. Wire YOLO detection → `encode_yolo_detections()` → `interaction_pomdp.update()`
2. Map POMDP actions to RC commands
3. Add interaction behaviors (wiggle, LED)

```python
# In main loop
if mode == "HUNTING":
    obs_token = encode_yolo_detections(yolo_results, ...)
    result = interaction_pomdp.update(obs_token)

    if result.action == "approach":
        cmd = compute_servoing(target_bbox, frame.shape)
        rc = (0, cmd.fb, cmd.ud, cmd.yaw)
    elif result.action == "interact_wiggle":
        rc = (0, 0, 0, 15 if frame_count % 20 < 10 else -15)
    elif result.action == "backoff":
        rc = (0, -30, 0, 0)
    # ...
```

#### Step 4: Human Search POMDP Integration (Priority: MEDIUM)

**File:** `pomdp/human_search.py` (already exists)

Connect to frontier explorer's room tracking:

```python
# When target detected
if target_detected:
    current_room = room_tracker.current_room
    human_search.update_sighting(current_room, confidence=0.9)

# When searching
if mode == "EXPLORATION":
    # Get search recommendation from human POMDP
    search_target = human_search.get_search_target()
    if search_target != current_room:
        # Bias frontier selection toward rooms where humans were seen
        frontier_explorer.set_preferred_direction(toward=search_target)
```

#### Step 5: Cat-Specific Behavior (Priority: LOW)

**File:** `cat_safe_shadow.py` (already exists)

Cat shadowing has different rules:
- Keep greater distance (3-4m)
- Avoid sudden movements
- Retreat if person detected (safety)
- Follow slowly if cat moves

```python
if target_class == "cat":
    target_distance = 3.5  # Farther than person
    max_approach_speed = 20  # Slower
    if person_also_detected:
        mode = "RETREAT"  # Safety first
```

### Testing Plan

#### Test 1: Mode Switching in Simulator
```bash
python test_hunting_pipeline.py --test mode_switch
```
- Spawn simulated person in room
- Verify: EXPLORATION → HUNTING transition
- Remove person, verify: HUNTING → EXPLORATION

#### Test 2: Visual Servoing
```bash
python test_hunting_pipeline.py --test servoing
```
- Place target at various positions
- Verify: drone centers and approaches
- Test: target moves, drone follows

#### Test 3: Full Pipeline
```bash
python test_hunting_pipeline.py --test full
```
- Start in Kitchen
- Explore until person found
- Approach and interact
- Person leaves, return to exploration

#### Test 4: Real Drone (Ground Test)
```bash
python person_hunter_pomdp.py --ground-test
```
- Drone on ground (no takeoff)
- YOLO running on video
- Verify mode switches and RC commands (logged, not sent)

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `pomdp/visual_servoing.py` | CREATE | Proportional control for target approach |
| `test_hunting_pipeline.py` | CREATE | Integration tests for hunting |
| `test_full_pipeline.py` | MODIFY | Add mode switching and hunting integration |
| `person_hunter_pomdp.py` | MODIFY | Use new pipeline architecture |

### Success Criteria

1. **Exploration Phase**: Drone explores 3+ rooms, builds valid occupancy map
2. **Detection**: YOLO detects person/cat within 1 second of visibility
3. **Mode Switch**: Transition happens within 3 frames of detection
4. **Approach**: Drone centers target and reaches 2m distance within 10 seconds
5. **Interaction**: Drone maintains position, executes wiggle/LED for 5 seconds
6. **Recovery**: Returns to exploration within 2 seconds of target loss

### Timeline Estimate

| Step | Effort | Dependencies |
|------|--------|--------------|
| Mode switching | 2-3 hours | None |
| Visual servoing | 2-3 hours | Mode switching |
| Interaction integration | 3-4 hours | Visual servoing |
| Human search integration | 2-3 hours | Mode switching |
| Testing and tuning | 4-6 hours | All above |
| **Total** | **13-19 hours** | |

---

## Phase 16: Map-While-Hunt (Variational Occupancy)

### Current State

We're already halfway to "map-while-hunt": in the current pipeline `SpatialMapper.update(...)` is called every step (it's not gated by mode), so the occupancy grid keeps updating during hunting. The missing piece is: **make the map update explicitly probabilistic + schedule it under a compute budget**, so hunting doesn't starve mapping (and mapping uncertainty can feed back into both pursuit + replanning).

### Architecture: Two-Layer Mapping

Split into two layers:

**1. Fast Planning Map (current)**
- 2D occupancy grid used by A*/frontier
- Updated every control tick or at throttled rate
- Must be stable and cheap

**2. Belief Map (variational/Bayesian)**
- Maintains uncertainty (not just single uint8 "occupancy value")
- Produces derived planning grid: `p(occupied)` mean + "unknownness"
- Enables Active-Inference arbitration: information gain vs pursuit reward

This is the same principle as VBGS: streaming variational updates of a latent map rather than ad-hoc threshold painting. VBGS does this for Gaussian splats; we do it first for occupancy because it's trivial and extremely effective (classic probabilistic robotics).

### Why It Matters for Hunting

When approaching a person/cat, we still:
- Refine free space ahead (reduces "doorway hallucinations" reappearing)
- Keep room-transition evidence current
- Keep frontiers fresh so target-loss recovery is instantaneous

### Implementation: Beta-Bernoulli Belief Map

Current `OccupancyMap` is a uint8 grid with increment/decrement heuristics. It works, but it's not a clean Bayesian object.

**Recommended representation: Beta-Bernoulli per cell**

For each cell `c`:
- Occupancy is Bernoulli with parameter `p_c`
- Prior/posterior over `p_c` is `Beta(α_c, β_c)`
- Update with "evidence" increments (soft counts) from:
  - Depth obstacle hits (weak, noisy)
  - Free-space ray traversal (weak-to-moderate)
  - Collision-confirmed obstacles (very strong)

Then:
- Posterior mean: `E[p_c] = α_c / (α_c + β_c)`
- Uncertainty: `Var(p_c)` tells you what's still ambiguous

**Implementation in `utils/occupancy_map.py`:**

```python
@dataclass
class BeliefOccupancyMap:
    alpha: np.ndarray  # float32[H,W] - occupied evidence
    beta: np.ndarray   # float32[H,W] - free evidence
    locked: np.ndarray # bool[H,W] - collision-confirmed cells

    def __init__(self, width: int, height: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.alpha = np.full((height, width), prior_alpha, dtype=np.float32)
        self.beta = np.full((height, width), prior_beta, dtype=np.float32)
        self.locked = np.zeros((height, width), dtype=bool)

    def update_free(self, x: int, y: int, weight: float = 1.0):
        """Ray passed through this cell - evidence it's free."""
        if not self.locked[y, x]:
            self.beta[y, x] += weight

    def update_occupied(self, x: int, y: int, weight: float = 1.0):
        """Depth sensed obstacle here - weak evidence it's occupied."""
        if not self.locked[y, x]:
            self.alpha[y, x] += weight

    def lock_collision(self, x: int, y: int):
        """Collision confirmed - very strong evidence, lock cell."""
        self.alpha[y, x] += 100.0  # Strong evidence
        self.locked[y, x] = True

    @property
    def p_occupied(self) -> np.ndarray:
        """Posterior mean P(occupied)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def entropy(self) -> np.ndarray:
        """Entropy of Beta distribution (uncertainty)."""
        from scipy.stats import beta as beta_dist
        # Approximate: high when alpha ≈ beta, low when one dominates
        total = self.alpha + self.beta
        p = self.alpha / total
        return -p * np.log(p + 1e-10) - (1-p) * np.log(1-p + 1e-10)

    def to_costmap(self, lambda_occ: float = 10.0, lambda_uncert: float = 2.0) -> np.ndarray:
        """Generate A* costmap from belief."""
        p_occ = self.p_occupied
        ent = self.entropy
        # Base cost + occupancy penalty + uncertainty penalty (for exploration)
        cost = 1.0 + lambda_occ * p_occ + lambda_uncert * ent
        # Blocked if p_occ > threshold
        cost[p_occ > 0.7] = np.inf
        return cost
```

### Scheduling: Hunt Without Starving Mapping

During HUNT/APPROACH:
- Run depth-ray updates at lower rate (e.g., 5 Hz) or reduce rays from 60 → 20
- Always apply collision-confirmed updates immediately

```python
class MappingScheduler:
    def __init__(self, full_rate_hz: float = 30.0, hunt_rate_hz: float = 5.0):
        self.full_rate_hz = full_rate_hz
        self.hunt_rate_hz = hunt_rate_hz
        self.last_update = 0.0

    def should_update(self, mode: str, current_time: float) -> bool:
        rate = self.hunt_rate_hz if mode == "HUNTING" else self.full_rate_hz
        interval = 1.0 / rate
        if current_time - self.last_update >= interval:
            self.last_update = current_time
            return True
        return False

    def get_ray_count(self, mode: str) -> int:
        return 20 if mode == "HUNTING" else 60
```

### Behavior Arbitration: Chase + Map + Safe Navigation

Keep discrete mode machine but make action selection in HUNT/APPROACH a weighted blend:

**Action Sources:**
1. Visual servoing controller (centering + approach)
2. Local obstacle avoidance (already exists)
3. Global planner / frontier / "reacquire" planner (when target lost/occluded)

**Arbitration Rule (simple, robust):**

Compute target confidence score each tick:
- Based on YOLO confidence + bounding-box area stability + time-since-last-seen

Then:
- **High confidence**: Prioritize servoing, but clamp velocity/turn-rate using occupancy belief (don't push into unknown aggressively)
- **Medium confidence**: Servoing + cautious replanning (small moves, keep target in frame)
- **Low confidence**: Switch to reacquire (biased frontier search), mapping continues

This makes "keep mapping while hunting" automatic: **mapping is never a mode, it's a service**.

```python
def compute_target_confidence(yolo_conf: float, bbox_area: float, frames_since_seen: int) -> float:
    """Compute confidence score for arbitration."""
    base = yolo_conf
    # Larger bbox = closer = more confident
    area_factor = min(1.0, bbox_area / 0.3)  # Saturate at 30% of frame
    # Decay with time since last seen
    time_decay = 0.9 ** frames_since_seen
    return base * area_factor * time_decay

def arbitrate_action(confidence: float, servo_cmd, avoid_cmd, replan_cmd):
    """Blend actions based on confidence."""
    if confidence > 0.7:
        # High confidence: mostly servoing, some avoidance
        return blend(servo_cmd, avoid_cmd, weights=[0.8, 0.2])
    elif confidence > 0.3:
        # Medium: cautious blend
        return blend(servo_cmd, avoid_cmd, replan_cmd, weights=[0.5, 0.3, 0.2])
    else:
        # Low: reacquire mode
        return blend(replan_cmd, avoid_cmd, weights=[0.7, 0.3])
```

### A* Integration with Belief Map

A* should treat cells as blocked if `p_occ > τ_block` (e.g., 0.7), but also add soft cost for `p_occ` so it prefers safer corridors.

In exploration, add term that prefers high-entropy frontiers (information gain):

```python
def astar_cost(cell, belief_map, mode):
    p_occ = belief_map.p_occupied[cell]
    ent = belief_map.entropy[cell]

    if p_occ > 0.7:
        return float('inf')  # Blocked

    base_cost = 1.0
    occ_cost = 5.0 * p_occ  # Prefer lower occupancy probability

    if mode == "EXPLORATION":
        # Prefer high-entropy cells (information gain)
        info_bonus = -2.0 * ent
    else:
        # During hunting, avoid uncertainty
        info_bonus = 1.0 * ent

    return base_cost + occ_cost + info_bonus
```

### Reacquire Bias for Frontier Explorer

When target is lost:
1. Define "sighting prior" over map (last-seen position + decay + room-transition likelihood)
2. Frontier scoring becomes: `score = frontier_gain + λ * sighting_prior(frontier)`

This is the Human Search POMDP influencing exploration:

```python
def compute_sighting_prior(position, last_seen_pos, last_seen_time, room_tracker):
    """Prior probability target is near this position."""
    # Distance decay from last sighting
    dist = np.linalg.norm(np.array(position) - np.array(last_seen_pos))
    dist_factor = np.exp(-dist / 2.0)  # 2m decay constant

    # Time decay
    time_factor = 0.95 ** (current_time - last_seen_time)

    # Room transition factor (target more likely in connected rooms)
    room_factor = 1.0
    if room_tracker.current_room != last_seen_room:
        if rooms_are_adjacent(room_tracker.current_room, last_seen_room):
            room_factor = 0.7
        else:
            room_factor = 0.3

    return dist_factor * time_factor * room_factor
```

### Where VBGS Fits (Realistic Phasing)

VBGS (Variational Bayes Gaussian Splatting) is specifically about continual scene learning as variational inference over Gaussian splat parameters, avoiding catastrophic forgetting in streaming settings.

**Practical constraint for Tello:**
- Need reasonably stable camera poses (VO/SLAM quality)
- Multi-view integration (RGB, ideally depth or stereo)
- GPU budget

**Phased approach:**

| Phase | Description | Benefit |
|-------|-------------|---------|
| **A (now)** | Variational occupancy belief grid (Beta map) | Immediate doorway robustness + chase safety |
| **B (sim-first)** | VBGS as auxiliary world model in GLB sim | Stress-test map-while-hunt at richer fidelity |
| **C (optional)** | Real-drone VBGS offboard | Run on laptop GPU, stream frames + poses |

### Concrete Implementation Steps

| Step | Description | Files |
|------|-------------|-------|
| 0 | Confirm mapping runs in hunting (it does) | - |
| 1 | Add `BeliefOccupancyMap` with alpha/beta arrays | `utils/occupancy_map.py` |
| 2 | Replace `_update_cell()` with Bayesian weight updates | `utils/occupancy_map.py` |
| 3 | Feed belief map into A* with soft costs | `pomdp/frontier_explorer.py` |
| 4 | Throttle mapping compute during HUNT | `test_full_pipeline.py` |
| 5 | Add reacquire bias to frontier explorer | `pomdp/frontier_explorer.py` |

---

## Phase 17: Persistent Explore-Then-Hunt Runner ✓ IMPLEMENTED

### Overview

A drop-in script that enables persistent mapping across sessions without refactoring the pipeline.

**File:** `run_persistent_explore_then_hunt.py`

### What It Does

1. Creates a `FullPipelineSimulator(--frontier)`
2. Loads saved map + odometry if it exists
3. Runs EXPLORATION for N frames (or until progress in unknown space)
4. Saves state
5. Switches to HUNTING for M frames
6. Saves state again

On subsequent runs, it resumes from the saved map and does a short "top-up exploration" focused on frontiers adjacent to unknown, then hunts again.

### Usage

```bash
# First run (no saved state yet)
python run_persistent_explore_then_hunt.py --frontier --explore_frames 800 --hunt_frames 800

# Subsequent runs (auto-loads state)
python run_persistent_explore_then_hunt.py --frontier --resume_explore_frames 250 --hunt_frames 800

# Stop hunting early when target detected
python run_persistent_explore_then_hunt.py --stop_on_detect
```

### What Gets Persisted

| Data | Purpose |
|------|---------|
| `grid` | Occupancy values |
| `visit_count` | Confidence per cell |
| `trajectory` | Path history |
| `places` / `place_labels` | Named locations |
| `pose_x/y/yaw` | Odometry state |
| `room_*` | Room transition tracker state |

### Benefits

- **Persistent occupancy map** across sessions
- **Top-up exploration** on resume picks new frontiers (based on unknown adjacency)
- **Map updates during hunting** (mapper update always runs in main loop)
- **Early stopping** when unknown% drops or target detected

### Recommended Next Upgrades

#### 1. Persist Blocked-Edge Memory and Frontier Blacklist

So it doesn't re-try the same failure approach after restart.

```python
# Add to save_pipeline_state:
"blocked_edges": np.array(list(pipeline.frontier_explorer._blocked_edges.items()), dtype=object),
"blocked_targets": np.array(pipeline.frontier_explorer._blocked_targets, dtype=object),
```

#### 2. Persist Sighting Prior (Human Search POMDP)

On resume, exploration is biased toward rooms/corridors where targets were last seen.

```python
# Add to save_pipeline_state:
"sighting_counts": human_search.sighting_counts,
"last_seen_room": human_search.last_seen_room,
"last_seen_time": human_search.last_seen_time,
```

---

## Research References

- [Bio-Inspired Topological Autonomous Navigation with Active Inference](https://arxiv.org/html/2508.07267) - Ghent University / VERSES. Core approach for incremental topological mapping, localization via observation matching, EFE-guided exploration.

- [Active Inference for Robot Planning & Control](https://www.verses.ai/research-blog/why-learn-if-you-can-infer-active-inference-for-robot-planning-control) - VERSES AI. Hierarchical generative models, VBGS spatial perception, inference-based control.

- [pymdp: A Python library for active inference](https://github.com/infer-actively/pymdp) - Reference implementation for discrete POMDP active inference.

- [Robot navigation as hierarchical active inference](https://www.sciencedirect.com/science/article/abs/pii/S0893608021002021) - Hierarchical structure for navigation.

- [Clone-Structured Cognitive Graphs (CSCG)](https://github.com/vicariousinc/naturecomm_cscg) - Vicarious Inc. Learning cognitive maps that resolve perceptual aliasing via clone splitting, with hierarchical abstraction via community detection.

- [Variational Bayes Gaussian Splatting (VBGS)](https://github.com/hmishfaq/VBGS) - Continual 3D mapping via variational inference on Gaussian splats. Suitable for streaming drone video.
