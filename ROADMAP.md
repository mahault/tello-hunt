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

### Phase 8: Exploration Mode ✓
- [x] `pomdp/exploration_mode.py`:
  - `ExplorationModePOMDP` class - systematic environment mapping
  - VFE-based transition criteria (not arbitrary thresholds)
  - State machine: SCANNING, APPROACHING_FRONTIER, BACKTRACKING, TRANSITIONING
  - `update(obs, world_model, loc_result)` - update exploration state
  - `should_transition_to_hunt()` - VFE + variance based decision
  - VFE tracking via sliding window for stability detection
  - `get_vfe_stats()` - mean and variance for diagnostics
  - `exploration_action_to_rc_control()` - faster rotation/movement for exploration
  - `ExplorationResult` dataclass with VFE diagnostics
- [x] Mode switching in `person_hunter_pomdp.py`:
  - `POMDPController.mode` - "exploration" or "hunting"
  - `set_mode(mode, lock)` - manual mode switching
  - Automatic transition from exploration to hunting when VFE stabilizes
  - Keyboard controls: E = exploration, H = hunting
  - Mode-specific visualization overlay
- [x] Config constants in `pomdp/config.py`:
  - `EXPLORATION_VFE_WINDOW` - frames to track VFE history
  - `EXPLORATION_VFE_THRESHOLD` - mean VFE below this = well-modeled
  - `EXPLORATION_VFE_VARIANCE_THRESHOLD` - variance below this = stable
  - `EXPLORATION_EPISTEMIC_WEIGHT`, `EXPLORATION_PRAGMATIC_WEIGHT`
  - `EXPLORATION_SCAN_YAW`, `EXPLORATION_FORWARD_FB`
  - `EXPLORATION_STATES`, `N_EXPLORATION_STATES`

**Key insight:** VFE naturally captures "information gain" - when observations fit the model well (low VFE) and are stable (low variance), exploration is complete. This is more principled than arbitrary rotation counts or location thresholds.

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

**Completed Implementation (Dec 2024):**

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

## Critical Bug Fixes (Dec 2024)

All fixes implemented and ready for testing.

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

### Issue 2: Ellipsis in escape_actions list
**File:** `pomdp/exploration_mode.py`

**Problem:** List literally contains Python `...` (Ellipsis object):
```python
escape_actions = [4, 4, 4, 1, 3, 3, 3, 1, 2, 2...]  # ... is Ellipsis!
```

**Symptoms:**
- Weird stuck behavior
- Actions become invalid/unhandled

**Fix:** Replace with explicit int sequence.

### Issue 3: Depth normalization is per-frame (unstable collision detection)
**File:** `utils/depth_estimator.py`

**Problem:** Per-frame min/max normalization makes "blocked" threshold non-stationary:
```python
depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
```

**Symptoms:**
- "Blocked" fluctuates even in same corridor
- Single near pixel rescales entire map

**Fixes:**
- Use percentiles (e.g., 20th percentile) instead of global min/max
- Temporal EMA of collision score
- Debounce "blocked" for K consecutive frames

### Issue 4: Room classification can never predict "hallway"
**File:** `mapping/semantic_world_model.py`

**Problem A:** No positive evidence mapping to "hallway" in OBJECT_TO_ROOM.

**Problem B:** Object evidence accumulates forever per place - one "sink" detection biases place toward kitchen permanently.

**Symptoms:**
- Hallway GT → predicted kitchen/bedroom/unknown
- Deceptively high confidence

**Fixes:**
- Add hallway priors (negative evidence: few objects + corridor shape)
- Add recency/forgetting for objects (EMA decay)
- Gate votes by object stability (must be observed N times)

### Issue 5: ORB keyframe creation lacks spatial gating
**File:** `pomdp/place_recognizer.py`

**Problem:** New keyframe based on cooldown only, not spatial separation. No check for:
- Rotation delta
- Translation delta
- Scene novelty

**Symptoms:**
- Many places in one hallway
- Oscillation between small subset

**Fixes:**
- Require minimum viewpoint change (yaw delta) or translation
- Hysteresis band: create keyframe only after M consecutive low-match frames

### Issue 6: VBGS not loading
**Problem:** Missing `jaxtyping` dependency.

**Fix:** `pip install jaxtyping`

---

## Research References

- [Bio-Inspired Topological Autonomous Navigation with Active Inference](https://arxiv.org/html/2508.07267) - Ghent University / VERSES. Core approach for incremental topological mapping, localization via observation matching, EFE-guided exploration.

- [Active Inference for Robot Planning & Control](https://www.verses.ai/research-blog/why-learn-if-you-can-infer-active-inference-for-robot-planning-control) - VERSES AI. Hierarchical generative models, VBGS spatial perception, inference-based control.

- [pymdp: A Python library for active inference](https://github.com/infer-actively/pymdp) - Reference implementation for discrete POMDP active inference.

- [Robot navigation as hierarchical active inference](https://www.sciencedirect.com/science/article/abs/pii/S0893608021002021) - Hierarchical structure for navigation.

- [Clone-Structured Cognitive Graphs (CSCG)](https://github.com/vicariousinc/naturecomm_cscg) - Vicarious Inc. Learning cognitive maps that resolve perceptual aliasing via clone splitting, with hierarchical abstraction via community detection.

- [Variational Bayes Gaussian Splatting (VBGS)](https://github.com/hmishfaq/VBGS) - Continual 3D mapping via variational inference on Gaussian splats. Suitable for streaming drone video.
