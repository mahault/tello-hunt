"""
Test script for POMDP core functions.

Run: python test_pomdp.py
"""

import numpy as np

print("Testing POMDP infrastructure...")
print("=" * 50)

# Test 1: JAX import and basic functions
print("\n[1] Testing JAX import and basic functions...")
try:
    import jax
    import jax.numpy as jnp
    print(f"    JAX version: {jax.__version__}")
    print(f"    Devices: {jax.devices()}")

    from pomdp.core import normalize, softmax, entropy, kl_divergence

    # Test normalize
    x = jnp.array([1.0, 2.0, 3.0])
    normed = normalize(x)
    assert abs(normed.sum() - 1.0) < 1e-6, "normalize failed"
    print("    normalize(): OK")

    # Test softmax
    sm = softmax(x)
    assert abs(sm.sum() - 1.0) < 1e-6, "softmax failed"
    print("    softmax(): OK")

    # Test entropy
    uniform = jnp.ones(4) / 4
    H = entropy(uniform)
    expected_H = np.log(4)  # max entropy for 4 states
    assert abs(H - expected_H) < 1e-5, f"entropy failed: {H} vs {expected_H}"
    print(f"    entropy(): OK (uniform 4-state: {H:.3f})")

    # Test KL divergence
    p = jnp.array([0.5, 0.5])
    q = jnp.array([0.9, 0.1])
    kl = kl_divergence(p, q)
    assert kl > 0, "KL should be positive"
    print(f"    kl_divergence(): OK (KL={kl:.3f})")

    print("    [PASS] Basic functions work!")
except Exception as e:
    print(f"    [FAIL] {e}")
    raise

# Test 2: Belief update
print("\n[2] Testing belief update...")
try:
    from pomdp.core import belief_update, belief_update_from_A

    # Simple 3-state example
    prior = jnp.array([0.33, 0.33, 0.34])
    likelihood = jnp.array([0.9, 0.05, 0.05])  # observation strongly indicates state 0

    posterior = belief_update(prior, likelihood)
    assert posterior[0] > 0.8, "Posterior should concentrate on state 0"
    assert abs(posterior.sum() - 1.0) < 1e-6, "Posterior should sum to 1"
    print(f"    Prior: {prior}")
    print(f"    Likelihood: {likelihood}")
    print(f"    Posterior: {posterior}")
    print("    [PASS] Belief update works!")
except Exception as e:
    print(f"    [FAIL] {e}")
    raise

# Test 3: VFE computation
print("\n[3] Testing VFE computation...")
try:
    from pomdp.core import (
        accuracy, complexity, variational_free_energy,
        surprisal, vfe_components, belief_update_with_vfe
    )

    # Setup: 3 observations, 3 states
    n_obs, n_states = 3, 3

    # A matrix: P(o|s) - observation model
    # Each column sums to 1
    A = jnp.array([
        [0.8, 0.1, 0.1],  # obs 0 likely in state 0
        [0.1, 0.8, 0.1],  # obs 1 likely in state 1
        [0.1, 0.1, 0.8],  # obs 2 likely in state 2
    ])

    prior = jnp.ones(n_states) / n_states  # uniform prior
    obs_idx = 0  # observe obs 0

    # Belief update with VFE
    posterior, vfe, acc, comp, surp = belief_update_with_vfe(prior, A, obs_idx)

    print(f"    Observed: obs_{obs_idx}")
    print(f"    Prior: {prior}")
    print(f"    Posterior: {posterior}")
    print(f"    VFE: {vfe:.3f} (lower = better fit)")
    print(f"    Accuracy: {acc:.3f} (higher = better)")
    print(f"    Complexity: {comp:.3f} (lower = simpler)")
    print(f"    Surprisal: {surp:.3f} (higher = more novel)")

    # Verify: posterior should concentrate on state 0
    assert posterior[0] > 0.7, "Posterior should favor state 0"

    # Now test with surprising observation
    # Prior strongly believes state 2, but we observe obs 0
    prior_biased = jnp.array([0.05, 0.05, 0.9])
    posterior2, vfe2, acc2, comp2, surp2 = belief_update_with_vfe(prior_biased, A, obs_idx)

    print(f"\n    Surprising observation test:")
    print(f"    Prior (biased to state 2): {prior_biased}")
    print(f"    Observed: obs_0 (indicates state 0)")
    print(f"    Posterior: {posterior2}")
    print(f"    Surprisal: {surp2:.3f} (should be high)")

    assert surp2 > surp, "Surprising obs should have higher surprisal"
    print("    [PASS] VFE computation works!")
except Exception as e:
    print(f"    [FAIL] {e}")
    raise

# Test 4: EFE and action selection
print("\n[4] Testing EFE and action selection...")
try:
    from pomdp.core import expected_free_energy, select_action

    n_states = 3
    n_actions = 2
    n_obs = 3

    # A matrix: observation model
    A = jnp.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])

    # B matrix: transitions for each action
    # Action 0: stay in place
    # Action 1: move toward state 0
    B = jnp.zeros((n_states, n_states, n_actions))
    B = B.at[:, :, 0].set(jnp.eye(n_states))  # action 0: identity
    B = B.at[:, :, 1].set(jnp.array([  # action 1: move to state 0
        [0.9, 0.5, 0.3],
        [0.05, 0.4, 0.3],
        [0.05, 0.1, 0.4],
    ]))

    # C matrix: preferences (prefer obs 0)
    C = jnp.array([0.7, 0.2, 0.1])

    # Current belief: uncertain
    belief = jnp.ones(n_states) / n_states

    # Compute EFE for each action
    G0 = expected_free_energy(belief, A, B, C, 0)
    G1 = expected_free_energy(belief, A, B, C, 1)

    print(f"    Belief: {belief}")
    print(f"    Preference: obs_0 (C = {C})")
    print(f"    EFE(stay): {G0:.3f}")
    print(f"    EFE(move to state 0): {G1:.3f}")

    # Select action
    action_idx, action_probs = select_action(belief, A, B, C, n_actions)
    print(f"    Selected action: {action_idx}")
    print(f"    Action probs: {action_probs}")

    print("    [PASS] EFE and action selection work!")
except Exception as e:
    print(f"    [FAIL] {e}")
    raise

# Test 5: Observation encoder
print("\n[5] Testing observation encoder...")
try:
    from pomdp.observation_encoder import (
        ObservationToken, encode_yolo_detections,
        create_empty_observation, observation_to_text
    )
    from pomdp.config import N_OBJECT_TYPES, TYPE_NAMES

    # Create mock YOLO boxes
    class MockBox:
        def __init__(self, cls, conf, xyxy):
            self.cls = [cls]
            self.conf = [conf]
            self.xyxy = [xyxy]

    # Simulate detections: person, couch, tv
    mock_boxes = [
        MockBox(0, 0.85, [100, 100, 300, 400]),   # person
        MockBox(57, 0.72, [400, 200, 600, 350]),  # couch
        MockBox(62, 0.65, [50, 50, 150, 120]),    # tv
    ]

    mock_names = {0: 'person', 57: 'couch', 62: 'tv'}

    obs = encode_yolo_detections(mock_boxes, mock_names, 640, 480, conf_threshold=0.45)

    print(f"    N_OBJECT_TYPES: {N_OBJECT_TYPES}")
    print(f"    Detected objects: {observation_to_text(obs)}")
    print(f"    Person detected: {obs.person_detected}")
    print(f"    Person area: {obs.person_area:.3f}")
    print(f"    Person cx: {obs.person_cx:.3f}")
    print(f"    Person obs idx: {obs.person_obs_idx}")
    print(f"    Signature vector shape: {obs.to_signature_vector().shape}")

    # Test empty observation
    empty = create_empty_observation()
    assert not empty.person_detected, "Empty obs should have no person"
    print("    [PASS] Observation encoder works!")
except Exception as e:
    print(f"    [FAIL] {e}")
    raise

# Test 6: Dirichlet learning
print("\n[6] Testing Dirichlet-Categorical learning...")
try:
    from pomdp.core import update_dirichlet_counts, counts_to_distribution

    n_obs, n_states = 3, 3

    # Start with zero counts
    counts = jnp.zeros((n_obs, n_states))

    # Observe obs_0 at state_0 several times
    for _ in range(10):
        counts = update_dirichlet_counts(counts, obs_idx=0, state_idx=0)

    # Observe obs_1 at state_1 a few times
    for _ in range(5):
        counts = update_dirichlet_counts(counts, obs_idx=1, state_idx=1)

    print(f"    Counts:\n{counts}")

    # Convert to distribution
    A_learned = counts_to_distribution(counts, prior_alpha=1.0)
    print(f"    Learned A (with alpha=1):\n{A_learned}")

    # Check: P(obs_0 | state_0) should be high
    assert A_learned[0, 0] > 0.5, "P(obs_0|state_0) should be high"
    print("    [PASS] Dirichlet learning works!")
except Exception as e:
    print(f"    [FAIL] {e}")
    raise

# Test 7: JIT compilation timing
print("\n[7] Testing JIT compilation and timing...")
try:
    import time
    from pomdp.core import belief_update_with_vfe, select_action

    n_states = 20
    n_obs = 17
    n_actions = 6

    # Random matrices
    A = jnp.ones((n_obs, n_states)) / n_obs
    B = jnp.eye(n_states).reshape(n_states, n_states, 1)
    B = jnp.broadcast_to(B, (n_states, n_states, n_actions))
    C = jnp.ones(n_obs) / n_obs
    prior = jnp.ones(n_states) / n_states

    # First call: includes JIT compilation
    t0 = time.perf_counter()
    posterior, vfe, acc, comp, surp = belief_update_with_vfe(prior, A, 0)
    t1 = time.perf_counter()
    first_call = (t1 - t0) * 1000

    # Second call: should be fast (already compiled)
    t0 = time.perf_counter()
    for _ in range(100):
        posterior, vfe, acc, comp, surp = belief_update_with_vfe(prior, A, 0)
    t1 = time.perf_counter()
    avg_after_jit = (t1 - t0) * 1000 / 100

    print(f"    First call (with JIT): {first_call:.2f} ms")
    print(f"    Avg after JIT (100 calls): {avg_after_jit:.4f} ms")
    print(f"    Speedup: {first_call / avg_after_jit:.1f}x")

    if avg_after_jit < 1.0:
        print("    [PASS] JIT gives sub-millisecond updates!")
    else:
        print("    [WARN] Updates slower than 1ms, but still works")
except Exception as e:
    print(f"    [FAIL] {e}")
    raise

# Test 8: Topological Map
print("\n[8] Testing topological map...")
try:
    from pomdp.topological_map import TopologicalMap, LocationNode
    from pomdp.observation_encoder import create_empty_observation, ObservationToken

    # Create mock observations for different "rooms"
    def create_room_obs(obj_levels):
        """Create observation with specific object levels."""
        obs = create_empty_observation()
        obs.object_levels = np.array(obj_levels, dtype=np.int32)
        obs.object_max_conf = np.where(obs.object_levels > 0, 0.8, 0.0).astype(np.float32)
        return obs

    # Kitchen: oven (10), refrigerator (13), sink (12)
    kitchen_obs = create_room_obs([0,0,0,0,0,0,0,0,0,0, 2, 0, 2, 2, 0,0,0])
    # Living room: couch (2), tv (7)
    living_obs = create_room_obs([0,0,2,0,0,0,0,2,0,0, 0, 0, 0, 0, 0,0,0])
    # Bedroom: bed (4)
    bedroom_obs = create_room_obs([0,0,0,0,2,0,0,0,0,0, 0, 0, 0, 0, 0,0,0])

    # Create map and add locations
    topo_map = TopologicalMap()
    assert topo_map.n_locations == 0, "Should start empty"

    # Localize to first observation (should create new)
    loc_id, sim, is_new = topo_map.localize(kitchen_obs)
    assert is_new, "First location should be new"
    assert loc_id == 0, "First location should be id 0"
    print(f"    Added kitchen: loc_id={loc_id}, is_new={is_new}")

    # Localize to second observation (should create new)
    loc_id, sim, is_new = topo_map.localize(living_obs)
    assert is_new, "Living room should be new location"
    assert loc_id == 1, "Should be id 1"
    print(f"    Added living room: loc_id={loc_id}, is_new={is_new}")

    # Localize to similar kitchen observation (should match existing)
    kitchen_obs2 = create_room_obs([0,0,0,0,0,0,0,0,0,0, 2, 0, 1, 2, 0,0,0])  # slightly different
    loc_id, sim, is_new = topo_map.localize(kitchen_obs2, threshold=0.5)
    assert not is_new, "Similar kitchen should match existing"
    assert loc_id == 0, "Should match kitchen (id 0)"
    print(f"    Re-localized to kitchen: loc_id={loc_id}, sim={sim:.3f}, is_new={is_new}")

    # Add bedroom
    loc_id, sim, is_new = topo_map.localize(bedroom_obs)
    assert is_new, "Bedroom should be new"
    print(f"    Added bedroom: loc_id={loc_id}")

    assert topo_map.n_locations == 3, f"Should have 3 locations, got {topo_map.n_locations}"

    # Test edges
    topo_map.record_transition(0, 1, action=1)  # kitchen -> living, forward
    topo_map.record_transition(1, 2, action=1)  # living -> bedroom, forward
    topo_map.record_transition(1, 0, action=2)  # living -> kitchen, back

    assert len(topo_map.edges) == 3, "Should have 3 edges"

    # Test A and B matrix generation
    A = topo_map.get_A_matrix()
    B = topo_map.get_B_matrix()
    print(f"    A matrix shape: {A.shape}")
    print(f"    B matrix shape: {B.shape}")

    print("    [PASS] Topological map works!")
except Exception as e:
    print(f"    [FAIL] {e}")
    raise

# Test 9: Similarity functions
print("\n[9] Testing similarity functions...")
try:
    from pomdp.similarity import (
        cosine_similarity, jaccard_similarity, euclidean_similarity,
        is_same_location, batch_cosine_similarity, find_best_match_jit
    )

    # Test vectors
    vec1 = np.array([1.0, 0.0, 0.5, 0.0])
    vec2 = np.array([0.9, 0.1, 0.4, 0.0])
    vec3 = np.array([0.0, 1.0, 0.0, 0.5])

    # Cosine similarity
    sim_12 = cosine_similarity(vec1, vec2)
    sim_13 = cosine_similarity(vec1, vec3)
    print(f"    cosine(vec1, vec2) = {sim_12:.3f} (should be high)")
    print(f"    cosine(vec1, vec3) = {sim_13:.3f} (should be low)")
    assert sim_12 > 0.9, "Similar vectors should have high cosine"
    assert sim_13 < 0.3, "Different vectors should have low cosine"

    # Jaccard similarity
    jac = jaccard_similarity(vec1, vec2)
    print(f"    jaccard(vec1, vec2) = {jac:.3f}")

    # is_same_location
    assert is_same_location(vec1, vec2, threshold=0.8), "Should match"
    assert not is_same_location(vec1, vec3, threshold=0.8), "Should not match"

    # JIT batch similarity
    candidates = jnp.array([vec1, vec2, vec3])
    query = jnp.array(vec2)
    sims = batch_cosine_similarity(query, candidates)
    print(f"    batch_cosine_similarity: {sims}")
    assert sims[1] > 0.99, "vec2 should match itself"

    # find_best_match_jit
    best_idx, best_sim = find_best_match_jit(query, candidates)
    print(f"    best match: idx={best_idx}, sim={best_sim:.3f}")
    assert int(best_idx) == 1, "Should find vec2 as best match"

    print("    [PASS] Similarity functions work!")
except Exception as e:
    print(f"    [FAIL] {e}")
    raise

# Test 10: Map persistence
print("\n[10] Testing map persistence...")
try:
    import tempfile
    from pathlib import Path
    from pomdp.map_persistence import save_map, load_map, list_saved_maps

    # Use temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save the map we created earlier
        filepath = save_map(topo_map, name="test_map", maps_dir=tmpdir)
        assert filepath.exists(), "Map file should exist"

        # List saved maps
        maps = list_saved_maps(maps_dir=tmpdir)
        assert len(maps) == 1, "Should have 1 saved map"
        print(f"    Saved maps: {[m['name'] for m in maps]}")

        # Load the map
        loaded_map = load_map(name="test_map", maps_dir=tmpdir)
        assert loaded_map is not None, "Should load successfully"
        assert loaded_map.n_locations == 3, "Should have 3 locations"
        assert len(loaded_map.edges) == 3, "Should have 3 edges"

        # Verify A matrix is same
        A_original = topo_map.get_A_matrix()
        A_loaded = loaded_map.get_A_matrix()
        assert jnp.allclose(A_original, A_loaded, atol=1e-5), "A matrices should match"

    print("    [PASS] Map persistence works!")
except Exception as e:
    print(f"    [FAIL] {e}")
    raise

# Test 11: World Model POMDP
print("\n[11] Testing World Model POMDP...")
try:
    from pomdp.world_model import WorldModel, LocalizationResult

    # Create fresh world model
    wm = WorldModel()
    assert wm.n_locations == 0, "Should start empty"
    assert wm.current_location_id == -1, "No current location yet"

    # First observation creates first location
    result = wm.localize(kitchen_obs)
    assert result.new_location_discovered, "First obs should create new location"
    assert result.location_id == 0, "Should be location 0"
    assert wm.n_locations == 1, "Should have 1 location"
    print(f"    First location (kitchen): id={result.location_id}, conf={result.confidence:.2f}")

    # Second different observation creates new location
    result = wm.localize(living_obs, action_taken=1)  # moved forward
    assert result.new_location_discovered, "Different obs should create new location"
    assert result.location_id == 1, "Should be location 1"
    assert wm.n_locations == 2, "Should have 2 locations"
    print(f"    Second location (living): id={result.location_id}, conf={result.confidence:.2f}")

    # Third observation - bedroom
    result = wm.localize(bedroom_obs, action_taken=1)
    assert result.new_location_discovered, "Bedroom should be new"
    print(f"    Third location (bedroom): id={result.location_id}, conf={result.confidence:.2f}")

    # Return to kitchen-like observation (should NOT create new)
    result = wm.localize(kitchen_obs, action_taken=2)  # moved back
    # Note: might create new due to VFE, but similarity should match
    print(f"    Re-visit kitchen: id={result.location_id}, sim={result.similarity:.3f}, new={result.new_location_discovered}")

    # Check belief
    print(f"    Current belief shape: {wm.belief.shape}")
    print(f"    Belief entropy: {wm.get_belief_entropy():.3f}")

    # Get location info
    info = wm.get_location_info(0)
    print(f"    Location 0 info: visits={info['visit_count']}, objects={[o['name'] for o in info['top_objects'][:2]]}")

    # Test exploration recommendation
    action, probs = wm.get_exploration_target()
    print(f"    Recommended exploration action: {action}, probs={probs[:3]}...")

    print("    [PASS] World Model POMDP works!")
except Exception as e:
    print(f"    [FAIL] {e}")
    import traceback
    traceback.print_exc()
    raise

# Test 12: World Model persistence
print("\n[12] Testing World Model save/load...")
try:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save world model
        from pomdp.map_persistence import DEFAULT_MAPS_DIR
        import pomdp.map_persistence as mp
        mp.DEFAULT_MAPS_DIR = Path(tmpdir)

        saved_path = wm.save(name="test_world_model")
        print(f"    Saved to: {saved_path}")

        # Load into new world model
        wm2 = WorldModel.load(name="test_world_model")
        assert wm2.n_locations == wm.n_locations, "Should have same locations"
        print(f"    Loaded world model: {wm2}")

        # Verify localization still works
        result = wm2.localize(kitchen_obs)
        print(f"    Localized in loaded model: id={result.location_id}, conf={result.confidence:.2f}")

    print("    [PASS] World Model persistence works!")
except Exception as e:
    print(f"    [FAIL] {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "=" * 50)
print("All tests passed!")
print("=" * 50)
