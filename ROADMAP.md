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

### Phase 3: World Model POMDP (with topological map)
- [ ] `pomdp/world_model.py`:
  - `localize(observation)` - find current node or create new
  - `update_belief_over_locations()` (JIT) - soft belief over nodes
  - `learn_observation(node, observation)` - update A counts
  - `learn_transition(from_node, to_node, action)` - update B counts
  - `get_location_confidence()` - how sure are we of current location?

### Phase 4: Human Search POMDP
- [ ] `pomdp/human_search.py`:
  - P(person_obs | human_loc, drone_loc)
  - `update_human_prior_from_sighting()` - learn where humans appear
  - `update_human_belief()` (JIT)
  - `get_search_target()` - where to look next

### Phase 5: Interaction Mode POMDP
- [ ] `pomdp/interaction_mode.py`:
  - Engagement state transitions
  - `get_interaction_obs()` (JIT)
  - `update_interaction_belief()` (JIT)
  - `select_interaction_action()` - EFE minimization

### Phase 6: Safety Module
- [ ] `safety/overrides.py`:
  - Battery checks
  - Collision detection
  - Emergency overrides

### Phase 7: Integration
- [ ] `utils/frame_grabber.py` - extract from person_hunter_safe.py
- [ ] `person_hunter_pomdp.py` - full POMDP integration with learning
  - Load existing map or start fresh
  - Learn during flight
  - Save map on landing
- [ ] Visualization overlay with beliefs + learned locations

### Phase 8: Exploration Mode
- [ ] Add dedicated exploration behavior:
  - Systematic room scanning when map is sparse
  - Prefer unexplored areas (information gain)
  - Build map before hunting

### Phase 9: Testing & Iteration
- [ ] Test map learning with stationary drone + manual carry
- [ ] Test location inference accuracy
- [ ] Flight testing with learning enabled
- [ ] Verify map persistence across sessions

---

## Dependencies

Add to environment.yml:
```yaml
- pip:
    - jax[cpu]>=0.4.20    # or jax[cuda12] for GPU
    - jaxlib>=0.4.20
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `pomdp/__init__.py` | Package init |
| `pomdp/config.py` | Constants, thresholds, N_MAX_LOCATIONS |
| `pomdp/core.py` | JAX JIT belief functions |
| `pomdp/observation_encoder.py` | YOLO → fixed observation tokens |
| `pomdp/topological_map.py` | TopologicalMap class, LocationNode |
| `pomdp/similarity.py` | Observation similarity for localization |
| `pomdp/map_persistence.py` | Save/load learned environment maps |
| `pomdp/world_model.py` | Location belief POMDP + learning |
| `pomdp/human_search.py` | Human location POMDP |
| `pomdp/interaction_mode.py` | Action selection POMDP |
| `pomdp/priors.py` | Context-based priors |
| `safety/__init__.py` | Package init |
| `safety/overrides.py` | Safety checks |
| `utils/__init__.py` | Package init |
| `utils/frame_grabber.py` | Extract from existing |
| `maps/` | Directory for saved learned maps |
| `person_hunter_pomdp.py` | New main script with learning |

## Files to Modify

| File | Change |
|------|--------|
| `environment.yml` | Add JAX dependencies |

## Reference Files (unchanged)

| File | Why |
|------|-----|
| `person_hunter_safe.py` | Reference for main loop (lines 216-402), YOLO integration, RC control |

---

## Research References

- [Bio-Inspired Topological Autonomous Navigation with Active Inference](https://arxiv.org/html/2508.07267) - Ghent University / VERSES. Core approach for incremental topological mapping, localization via observation matching, EFE-guided exploration.

- [Active Inference for Robot Planning & Control](https://www.verses.ai/research-blog/why-learn-if-you-can-infer-active-inference-for-robot-planning-control) - VERSES AI. Hierarchical generative models, VBGS spatial perception, inference-based control.

- [pymdp: A Python library for active inference](https://github.com/infer-actively/pymdp) - Reference implementation for discrete POMDP active inference.

- [Robot navigation as hierarchical active inference](https://www.sciencedirect.com/science/article/abs/pii/S0893608021002021) - Hierarchical structure for navigation.
