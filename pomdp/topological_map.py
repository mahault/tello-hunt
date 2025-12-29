"""
Topological map for learned environment representation.

Based on Bio-Inspired Topological Navigation with Active Inference.
The map grows incrementally as the drone explores, with nodes representing
distinct locations and edges representing traversable transitions.
"""

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .config import (
    N_MAX_LOCATIONS, N_OBJECT_TYPES, N_OBS_LEVELS, N_MOVEMENT_ACTIONS,
    LOCATION_SIMILARITY_THRESHOLD, MIN_VISITS_FOR_ESTABLISHED,
    DIRICHLET_PRIOR_ALPHA
)
from .observation_encoder import ObservationToken


@dataclass
class LocationNode:
    """
    A node in the topological map representing a distinct location.

    Locations are identified by their observation signatures (what objects
    are visible from that location) and optionally by image embeddings.
    """
    # Unique identifier
    id: int

    # Running mean of observation signatures seen at this location
    # Shape: (N_OBJECT_TYPES,)
    observation_signature: np.ndarray

    # Count of each observation level seen at this location
    # Shape: (N_OBJECT_TYPES, N_OBS_LEVELS) for learning A matrix
    # A_counts[obj_type, obs_level] = count
    A_counts: np.ndarray

    # Number of times this location has been visited
    visit_count: int = 1

    # Running sum for online mean update
    _signature_sum: np.ndarray = field(default=None, repr=False)

    # Image embedding for CLIP-based localization (optional)
    # Shape: (512,) for CLIP ViT-B/32
    image_embedding: np.ndarray = field(default=None)
    _embedding_sum: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize derived fields."""
        if self._signature_sum is None:
            self._signature_sum = self.observation_signature.copy()
        if self.image_embedding is not None and self._embedding_sum is None:
            self._embedding_sum = self.image_embedding.copy()

    def update_signature(self, new_obs: np.ndarray):
        """
        Update the location signature with a new observation.

        Uses online mean update for numerical stability.
        """
        self.visit_count += 1
        self._signature_sum = self._signature_sum + new_obs
        self.observation_signature = self._signature_sum / self.visit_count

    def update_embedding(self, new_embedding: np.ndarray):
        """
        Update the image embedding with a new observation.

        Uses online mean update with re-normalization.
        """
        if self.image_embedding is None:
            self.image_embedding = new_embedding.copy()
            self._embedding_sum = new_embedding.copy()
        else:
            self._embedding_sum = self._embedding_sum + new_embedding
            self.image_embedding = self._embedding_sum / self.visit_count
            # Re-normalize
            norm = np.linalg.norm(self.image_embedding)
            if norm > 0:
                self.image_embedding = self.image_embedding / norm

    def update_A_counts(self, object_levels: np.ndarray):
        """
        Update observation counts for learning A matrix.

        Args:
            object_levels: Shape (N_OBJECT_TYPES,), values 0, 1, or 2
        """
        for obj_type, level in enumerate(object_levels):
            self.A_counts[obj_type, int(level)] += 1

    def is_established(self) -> bool:
        """Check if this location has been visited enough times."""
        return self.visit_count >= MIN_VISITS_FOR_ESTABLISHED

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON persistence."""
        data = {
            'id': self.id,
            'observation_signature': self.observation_signature.tolist(),
            'A_counts': self.A_counts.tolist(),
            'visit_count': self.visit_count,
            '_signature_sum': self._signature_sum.tolist(),
        }
        if self.image_embedding is not None:
            data['image_embedding'] = self.image_embedding.tolist()
            data['_embedding_sum'] = self._embedding_sum.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'LocationNode':
        """Deserialize from dictionary."""
        node = cls(
            id=data['id'],
            observation_signature=np.array(data['observation_signature'], dtype=np.float32),
            A_counts=np.array(data['A_counts'], dtype=np.float32),
            visit_count=data['visit_count'],
        )
        node._signature_sum = np.array(data['_signature_sum'], dtype=np.float32)
        if 'image_embedding' in data:
            node.image_embedding = np.array(data['image_embedding'], dtype=np.float32)
            node._embedding_sum = np.array(data['_embedding_sum'], dtype=np.float32)
        return node


@dataclass
class Edge:
    """
    An edge in the topological map representing a transition between locations.
    """
    from_node_id: int
    to_node_id: int
    action: int  # Movement action index

    # Number of times this transition was observed
    count: int = 1

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'from_node_id': self.from_node_id,
            'to_node_id': self.to_node_id,
            'action': self.action,
            'count': self.count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Edge':
        """Deserialize from dictionary."""
        return cls(
            from_node_id=data['from_node_id'],
            to_node_id=data['to_node_id'],
            action=data['action'],
            count=data['count'],
        )


class TopologicalMap:
    """
    Learned topological map of the environment.

    The map consists of:
    - Nodes: Distinct locations identified by observation signatures
    - Edges: Transitions between locations with action labels

    The map grows as the drone explores, and can be used to:
    - Localize: Find which known location matches current observation
    - Plan: Use edge structure for navigation
    - Learn: Update A and B matrices from experience
    """

    def __init__(self):
        """Initialize empty map."""
        self.nodes: List[LocationNode] = []
        self.edges: List[Edge] = []

        # Quick lookup: edge key (from, to, action) -> edge index
        self._edge_lookup: Dict[Tuple[int, int, int], int] = {}

        # Current location ID (-1 if unknown)
        self.current_location_id: int = -1

    @property
    def n_locations(self) -> int:
        """Number of locations in the map."""
        return len(self.nodes)

    def add_node(self, observation: ObservationToken) -> LocationNode:
        """
        Add a new location node from an observation.

        Args:
            observation: Current observation token

        Returns:
            The newly created node
        """
        if self.n_locations >= N_MAX_LOCATIONS:
            raise ValueError(f"Maximum locations ({N_MAX_LOCATIONS}) reached")

        node_id = self.n_locations
        signature = observation.to_signature_vector()

        # Initialize A_counts from first observation
        A_counts = np.zeros((N_OBJECT_TYPES, N_OBS_LEVELS), dtype=np.float32)
        for obj_type, level in enumerate(observation.object_levels):
            A_counts[obj_type, int(level)] = 1.0

        node = LocationNode(
            id=node_id,
            observation_signature=signature,
            A_counts=A_counts,
            visit_count=1,
        )
        self.nodes.append(node)
        return node

    def get_node(self, node_id: int) -> Optional[LocationNode]:
        """Get node by ID."""
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id]
        return None

    def find_best_match(
        self,
        observation: ObservationToken,
        similarity_func: callable = None
    ) -> Tuple[Optional[int], float]:
        """
        Find the best matching location for an observation.

        Args:
            observation: Current observation token
            similarity_func: Similarity function (default: cosine similarity)

        Returns:
            (node_id, similarity) if match found, (-1, 0.0) otherwise
        """
        if len(self.nodes) == 0:
            return -1, 0.0

        # Import here to avoid circular import
        from .similarity import cosine_similarity

        if similarity_func is None:
            similarity_func = cosine_similarity

        obs_signature = observation.to_signature_vector()

        best_id = -1
        best_sim = 0.0

        for node in self.nodes:
            sim = similarity_func(obs_signature, node.observation_signature)
            if sim > best_sim:
                best_sim = sim
                best_id = node.id

        return best_id, best_sim

    def add_node_with_embedding(
        self,
        observation: ObservationToken,
        embedding: np.ndarray
    ) -> LocationNode:
        """
        Add a new location node with an image embedding.

        Args:
            observation: Current observation token
            embedding: Image embedding from CLIP (512,)

        Returns:
            The newly created node
        """
        node = self.add_node(observation)
        node.image_embedding = embedding.copy()
        node._embedding_sum = embedding.copy()
        return node

    def find_best_match_embedding(
        self,
        embedding: np.ndarray
    ) -> Tuple[int, float]:
        """
        Find the best matching location using image embeddings.

        Args:
            embedding: Query image embedding (512,)

        Returns:
            (node_id, similarity) if match found, (-1, 0.0) otherwise
        """
        if len(self.nodes) == 0:
            return -1, 0.0

        # Only consider nodes that have embeddings
        nodes_with_embeddings = [n for n in self.nodes if n.image_embedding is not None]
        if len(nodes_with_embeddings) == 0:
            return -1, 0.0

        best_id = -1
        best_sim = 0.0

        for node in nodes_with_embeddings:
            # Cosine similarity (embeddings are normalized)
            sim = float(np.dot(embedding, node.image_embedding))
            if sim > best_sim:
                best_sim = sim
                best_id = node.id

        return best_id, best_sim

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[int]]:
        """
        Get all node embeddings as a matrix.

        Returns:
            (embeddings, node_ids) where embeddings is (N, 512)
        """
        nodes_with_embeddings = [n for n in self.nodes if n.image_embedding is not None]
        if len(nodes_with_embeddings) == 0:
            return np.empty((0, 512)), []

        embeddings = np.stack([n.image_embedding for n in nodes_with_embeddings])
        node_ids = [n.id for n in nodes_with_embeddings]
        return embeddings, node_ids

    def localize(
        self,
        observation: ObservationToken,
        threshold: float = LOCATION_SIMILARITY_THRESHOLD
    ) -> Tuple[int, float, bool]:
        """
        Localize the drone based on current observation.

        Args:
            observation: Current observation token
            threshold: Minimum similarity to consider a match

        Returns:
            (location_id, similarity, is_new_location)
            If is_new_location is True, a new node was added.
        """
        best_id, best_sim = self.find_best_match(observation)

        if best_sim >= threshold and best_id >= 0:
            # Match found - update existing node
            node = self.nodes[best_id]
            node.update_signature(observation.to_signature_vector())
            node.update_A_counts(observation.object_levels)
            self.current_location_id = best_id
            return best_id, best_sim, False
        else:
            # No match - add new location
            node = self.add_node(observation)
            self.current_location_id = node.id
            return node.id, 1.0, True

    def add_edge(
        self,
        from_node_id: int,
        to_node_id: int,
        action: int
    ) -> Edge:
        """
        Add or update an edge representing a transition.

        Args:
            from_node_id: Source location ID
            to_node_id: Destination location ID
            action: Movement action that caused the transition

        Returns:
            The edge (new or updated)
        """
        key = (from_node_id, to_node_id, action)

        if key in self._edge_lookup:
            # Update existing edge
            edge_idx = self._edge_lookup[key]
            self.edges[edge_idx].count += 1
            return self.edges[edge_idx]
        else:
            # Create new edge
            edge = Edge(
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                action=action,
                count=1,
            )
            self._edge_lookup[key] = len(self.edges)
            self.edges.append(edge)
            return edge

    def record_transition(
        self,
        from_location: int,
        to_location: int,
        action: int
    ):
        """
        Record a transition from one location to another.

        Call this after successfully moving and re-localizing.
        """
        if from_location >= 0 and to_location >= 0:
            self.add_edge(from_location, to_location, action)

    def get_A_matrix(self, prior_alpha: float = DIRICHLET_PRIOR_ALPHA) -> jnp.ndarray:
        """
        Compute A matrix P(observation | location) from counts.

        Each column sums to 1.

        Returns:
            Shape: (N_OBJECT_TYPES * N_OBS_LEVELS, n_locations)
            Flattened observation indices.
        """
        n_locs = max(1, self.n_locations)
        n_obs = N_OBJECT_TYPES * N_OBS_LEVELS

        # Initialize with prior
        A = np.ones((n_obs, n_locs), dtype=np.float32) * prior_alpha

        for node in self.nodes:
            # Flatten A_counts from (N_OBJECT_TYPES, N_OBS_LEVELS) to (n_obs,)
            flat_counts = node.A_counts.flatten()
            A[:, node.id] = flat_counts + prior_alpha

        # Normalize columns
        A = A / A.sum(axis=0, keepdims=True)

        return jnp.array(A)

    def get_B_matrix(self, prior_alpha: float = DIRICHLET_PRIOR_ALPHA) -> jnp.ndarray:
        """
        Compute B matrix P(location' | location, action) from transition counts.

        Returns:
            Shape: (n_locations, n_locations, N_MOVEMENT_ACTIONS)
        """
        n_locs = max(1, self.n_locations)

        # Initialize with prior (slight self-loop bias)
        B = np.ones((n_locs, n_locs, N_MOVEMENT_ACTIONS), dtype=np.float32) * prior_alpha

        # Add self-loops for "stay" action (action 0)
        for i in range(n_locs):
            B[i, i, 0] += 5.0  # Strong prior for staying in place

        # Add observed transitions
        for edge in self.edges:
            if edge.from_node_id < n_locs and edge.to_node_id < n_locs:
                B[edge.to_node_id, edge.from_node_id, edge.action] += edge.count

        # Normalize: for each (from_loc, action), probabilities over to_loc sum to 1
        for action in range(N_MOVEMENT_ACTIONS):
            B[:, :, action] = B[:, :, action] / B[:, :, action].sum(axis=0, keepdims=True)

        return jnp.array(B)

    def get_neighbors(self, node_id: int) -> List[Tuple[int, int, int]]:
        """
        Get all neighbors of a node with edge info.

        Returns:
            List of (neighbor_id, action, count) tuples
        """
        neighbors = []
        for edge in self.edges:
            if edge.from_node_id == node_id:
                neighbors.append((edge.to_node_id, edge.action, edge.count))
        return neighbors

    def get_all_signatures(self) -> np.ndarray:
        """
        Get all location signatures as a matrix.

        Returns:
            Shape: (n_locations, N_OBJECT_TYPES)
        """
        if len(self.nodes) == 0:
            return np.zeros((0, N_OBJECT_TYPES), dtype=np.float32)

        return np.stack([n.observation_signature for n in self.nodes])

    def to_dict(self) -> Dict:
        """Serialize entire map to dictionary for JSON persistence."""
        return {
            'nodes': [n.to_dict() for n in self.nodes],
            'edges': [e.to_dict() for e in self.edges],
            'current_location_id': self.current_location_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TopologicalMap':
        """Deserialize map from dictionary."""
        topo_map = cls()
        topo_map.nodes = [LocationNode.from_dict(n) for n in data['nodes']]
        topo_map.edges = [Edge.from_dict(e) for e in data['edges']]
        topo_map.current_location_id = data.get('current_location_id', -1)

        # Rebuild edge lookup
        for i, edge in enumerate(topo_map.edges):
            key = (edge.from_node_id, edge.to_node_id, edge.action)
            topo_map._edge_lookup[key] = i

        return topo_map

    def __repr__(self) -> str:
        return f"TopologicalMap(n_locations={self.n_locations}, n_edges={len(self.edges)})"
