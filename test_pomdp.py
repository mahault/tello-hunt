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

print("\n" + "=" * 50)
print("All tests passed!")
print("=" * 50)
