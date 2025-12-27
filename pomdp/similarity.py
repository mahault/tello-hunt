"""
Observation similarity metrics for topological localization.

These functions compare observation signatures to determine if two
observations come from the same location.
"""

import numpy as np
import jax.numpy as jnp
import jax
from typing import Union

from .config import LOCATION_SIMILARITY_THRESHOLD, N_OBJECT_TYPES


def cosine_similarity(
    vec1: Union[np.ndarray, jnp.ndarray],
    vec2: Union[np.ndarray, jnp.ndarray],
    eps: float = 1e-8
) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector
        eps: Small constant to avoid division by zero

    Returns:
        Cosine similarity in range [-1, 1]
    """
    # Handle both numpy and jax arrays
    if isinstance(vec1, jnp.ndarray):
        dot = jnp.dot(vec1, vec2)
        norm1 = jnp.linalg.norm(vec1)
        norm2 = jnp.linalg.norm(vec2)
        return float(dot / (norm1 * norm2 + eps))
    else:
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return float(dot / (norm1 * norm2 + eps))


def weighted_cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    weights: np.ndarray = None,
    eps: float = 1e-8
) -> float:
    """
    Compute weighted cosine similarity.

    Allows emphasizing certain object types (e.g., distinctive furniture
    over common objects like chairs).

    Args:
        vec1: First vector
        vec2: Second vector
        weights: Per-element weights (default: uniform)
        eps: Small constant to avoid division by zero

    Returns:
        Weighted cosine similarity in range [-1, 1]
    """
    if weights is None:
        return cosine_similarity(vec1, vec2, eps)

    w_vec1 = vec1 * weights
    w_vec2 = vec2 * weights

    dot = np.dot(w_vec1, w_vec2)
    norm1 = np.linalg.norm(w_vec1)
    norm2 = np.linalg.norm(w_vec2)

    return float(dot / (norm1 * norm2 + eps))


def jaccard_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Compute Jaccard similarity (intersection over union) of detected objects.

    Treats vectors as sets of detected objects.

    Args:
        vec1: First signature vector
        vec2: Second signature vector
        threshold: Minimum value to consider object "present"

    Returns:
        Jaccard similarity in range [0, 1]
    """
    set1 = vec1 > threshold
    set2 = vec2 > threshold

    intersection = np.sum(set1 & set2)
    union = np.sum(set1 | set2)

    if union == 0:
        return 1.0  # Both empty = same

    return float(intersection / union)


def euclidean_distance(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:
    """
    Compute Euclidean distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance (lower = more similar)
    """
    return float(np.linalg.norm(vec1 - vec2))


def euclidean_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    scale: float = 1.0
) -> float:
    """
    Convert Euclidean distance to similarity score.

    Args:
        vec1: First vector
        vec2: Second vector
        scale: Scale factor (higher = broader similarity)

    Returns:
        Similarity in range [0, 1]
    """
    dist = euclidean_distance(vec1, vec2)
    return float(np.exp(-dist / scale))


def is_same_location(
    obs_signature: np.ndarray,
    node_signature: np.ndarray,
    threshold: float = LOCATION_SIMILARITY_THRESHOLD,
    similarity_func: callable = None
) -> bool:
    """
    Determine if an observation matches a known location.

    Args:
        obs_signature: Current observation signature
        node_signature: Stored location signature
        threshold: Minimum similarity to be considered the same
        similarity_func: Similarity function (default: cosine)

    Returns:
        True if observation matches the location
    """
    if similarity_func is None:
        similarity_func = cosine_similarity

    sim = similarity_func(obs_signature, node_signature)
    return sim >= threshold


def find_most_similar(
    query: np.ndarray,
    candidates: np.ndarray,
    similarity_func: callable = None
) -> tuple:
    """
    Find the most similar candidate to a query.

    Args:
        query: Query vector shape (N_OBJECT_TYPES,)
        candidates: Matrix of candidate vectors shape (n_candidates, N_OBJECT_TYPES)
        similarity_func: Similarity function (default: cosine)

    Returns:
        (best_index, best_similarity)
    """
    if similarity_func is None:
        similarity_func = cosine_similarity

    if len(candidates) == 0:
        return -1, 0.0

    similarities = [similarity_func(query, c) for c in candidates]
    best_idx = int(np.argmax(similarities))
    return best_idx, similarities[best_idx]


# =============================================================================
# JIT-compiled similarity for batch operations
# =============================================================================

@jax.jit
def batch_cosine_similarity(
    query: jnp.ndarray,
    candidates: jnp.ndarray,
    eps: float = 1e-8
) -> jnp.ndarray:
    """
    Compute cosine similarity of query against all candidates (JIT compiled).

    Args:
        query: Query vector shape (N_OBJECT_TYPES,)
        candidates: Matrix of candidate vectors shape (n_candidates, N_OBJECT_TYPES)
        eps: Small constant to avoid division by zero

    Returns:
        Similarities shape (n_candidates,)
    """
    # Normalize query
    query_norm = query / (jnp.linalg.norm(query) + eps)

    # Normalize candidates (each row)
    cand_norms = jnp.linalg.norm(candidates, axis=1, keepdims=True) + eps
    cand_normalized = candidates / cand_norms

    # Dot products
    similarities = jnp.dot(cand_normalized, query_norm)

    return similarities


@jax.jit
def find_best_match_jit(
    query: jnp.ndarray,
    candidates: jnp.ndarray,
    eps: float = 1e-8
) -> tuple:
    """
    Find best matching candidate using JIT (for real-time localization).

    Args:
        query: Query vector shape (N_OBJECT_TYPES,)
        candidates: Matrix of candidate vectors shape (n_candidates, N_OBJECT_TYPES)
        eps: Small constant to avoid division by zero

    Returns:
        (best_index, best_similarity)
    """
    similarities = batch_cosine_similarity(query, candidates, eps)
    best_idx = jnp.argmax(similarities)
    best_sim = similarities[best_idx]
    return best_idx, best_sim


# =============================================================================
# Distinctive object weights (optional)
# =============================================================================

# Objects that are more distinctive for room identification
# Higher weight = more important for location matching
DEFAULT_OBJECT_WEIGHTS = np.array([
    0.5,   # person - not useful for location
    0.7,   # chair - somewhat common
    1.2,   # couch - distinctive
    0.8,   # potted_plant - moderate
    1.5,   # bed - very distinctive (bedroom)
    1.0,   # dining_table - distinctive
    1.5,   # toilet - very distinctive (bathroom)
    1.2,   # tv - distinctive (living room)
    0.8,   # laptop - mobile, less useful
    0.3,   # cell_phone - mobile, not useful
    1.5,   # oven - very distinctive (kitchen)
    1.3,   # toaster - distinctive (kitchen)
    1.2,   # sink - distinctive (kitchen/bathroom)
    1.5,   # refrigerator - very distinctive (kitchen)
    0.5,   # book - common, mobile
    0.8,   # clock - moderate
    1.3,   # toothbrush - distinctive (bathroom)
], dtype=np.float32)


def get_distinctive_weights() -> np.ndarray:
    """Get default object weights for weighted similarity."""
    return DEFAULT_OBJECT_WEIGHTS.copy()
