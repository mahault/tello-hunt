"""
Embedding tokenizer for CSCG.

Converts continuous image embeddings to discrete observation tokens
via online k-means clustering.

Supports multiple encoders:
- DINOv2: 384 dimensions (default, better for place recognition)
- CLIP: 512 dimensions
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class TokenizerConfig:
    """Configuration for embedding tokenizer."""
    n_tokens: int = 32  # Maximum number of observation tokens
    embedding_dim: int = 384  # DINOv2-small default (512 for CLIP)
    similarity_threshold: float = 0.85  # Cosine similarity to assign existing token
    min_samples_per_token: int = 5  # Minimum samples before token is "established"


class EmbeddingTokenizer:
    """
    Online tokenizer that clusters CLIP embeddings into discrete tokens.

    Uses a streaming k-means-like approach:
    - If embedding is similar to existing centroid -> assign that token
    - If embedding is novel (low similarity to all) -> create new token

    Attributes:
        centroids: Token centroids (n_tokens, embedding_dim)
        counts: Sample counts per token (n_tokens,)
        n_active: Number of active tokens
    """

    def __init__(
        self,
        n_tokens: int = 32,
        embedding_dim: int = 384,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize tokenizer.

        Args:
            n_tokens: Maximum number of tokens
            embedding_dim: Embedding dimension (384 for DINOv2-small, 512 for CLIP)
            similarity_threshold: Cosine similarity threshold for assignment
        """
        self.n_tokens = n_tokens
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold

        # Token centroids (running mean of assigned embeddings)
        self.centroids = np.zeros((n_tokens, embedding_dim), dtype=np.float32)

        # Sample counts per token
        self.counts = np.zeros(n_tokens, dtype=np.int32)

        # Running sum for online mean update
        self._centroid_sums = np.zeros((n_tokens, embedding_dim), dtype=np.float64)

        # Number of active (used) tokens
        self.n_active = 0

    def tokenize(self, embedding: np.ndarray) -> Tuple[int, float]:
        """
        Convert embedding to discrete token.

        Args:
            embedding: CLIP embedding (embedding_dim,)

        Returns:
            token: Token index
            similarity: Cosine similarity to assigned centroid
        """
        embedding = np.asarray(embedding, dtype=np.float32).flatten()

        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        if self.n_active == 0:
            # First token
            return self._add_token(embedding), 1.0

        # Compute similarity to all active centroids
        similarities = self._compute_similarities(embedding)

        best_token = np.argmax(similarities)
        best_sim = similarities[best_token]

        if best_sim >= self.similarity_threshold:
            # Assign to existing token and update centroid
            self._update_centroid(best_token, embedding)
            return best_token, best_sim
        elif self.n_active < self.n_tokens:
            # Create new token
            return self._add_token(embedding), 1.0
        else:
            # Max tokens reached - assign to nearest anyway
            self._update_centroid(best_token, embedding)
            return best_token, best_sim

    def tokenize_batch(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize a batch of embeddings.

        Args:
            embeddings: (batch_size, embedding_dim)

        Returns:
            tokens: (batch_size,) token indices
            similarities: (batch_size,) similarities
        """
        tokens = np.zeros(len(embeddings), dtype=np.int32)
        similarities = np.zeros(len(embeddings), dtype=np.float32)

        for i, emb in enumerate(embeddings):
            tokens[i], similarities[i] = self.tokenize(emb)

        return tokens, similarities

    def get_token(self, embedding: np.ndarray) -> Tuple[int, float]:
        """
        Get token for embedding without updating centroids (inference only).

        Args:
            embedding: CLIP embedding

        Returns:
            token: Token index
            similarity: Cosine similarity
        """
        embedding = np.asarray(embedding, dtype=np.float32).flatten()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        if self.n_active == 0:
            return 0, 0.0

        similarities = self._compute_similarities(embedding)
        best_token = np.argmax(similarities)
        return best_token, similarities[best_token]

    def _compute_similarities(self, embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarities to all active centroids."""
        similarities = np.zeros(self.n_active, dtype=np.float32)

        for i in range(self.n_active):
            # Centroids are already normalized
            similarities[i] = np.dot(embedding, self.centroids[i])

        return similarities

    def _add_token(self, embedding: np.ndarray) -> int:
        """Add a new token with given embedding as centroid."""
        token = self.n_active
        self.centroids[token] = embedding
        self._centroid_sums[token] = embedding.astype(np.float64)
        self.counts[token] = 1
        self.n_active += 1
        return token

    def _update_centroid(self, token: int, embedding: np.ndarray):
        """Update centroid with new embedding (online mean)."""
        self.counts[token] += 1
        self._centroid_sums[token] += embedding.astype(np.float64)

        # Recompute mean and normalize
        mean = self._centroid_sums[token] / self.counts[token]
        norm = np.linalg.norm(mean)
        if norm > 0:
            self.centroids[token] = (mean / norm).astype(np.float32)

    def fit(self, embeddings: np.ndarray, n_iter: int = 10):
        """
        Batch fit tokenizer using k-means.

        Args:
            embeddings: (n_samples, embedding_dim)
            n_iter: Number of k-means iterations
        """
        from sklearn.cluster import KMeans

        n_samples = len(embeddings)
        n_clusters = min(self.n_tokens, n_samples)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings_norm = embeddings / norms

        # Run k-means
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=n_iter, random_state=42)
        labels = kmeans.fit_predict(embeddings_norm)

        # Store centroids (normalized)
        for i in range(n_clusters):
            centroid = kmeans.cluster_centers_[i]
            norm = np.linalg.norm(centroid)
            if norm > 0:
                self.centroids[i] = (centroid / norm).astype(np.float32)
            self.counts[i] = np.sum(labels == i)
            self._centroid_sums[i] = centroid * self.counts[i]

        self.n_active = n_clusters

    def get_token_info(self, token: int) -> dict:
        """Get information about a token."""
        if token >= self.n_active:
            return {'error': 'Invalid token'}

        return {
            'token': token,
            'count': int(self.counts[token]),
            'centroid_norm': float(np.linalg.norm(self.centroids[token])),
        }

    def save(self, filepath: str):
        """Save tokenizer state."""
        np.savez(
            filepath,
            centroids=self.centroids,
            counts=self.counts,
            centroid_sums=self._centroid_sums,
            n_active=self.n_active,
            n_tokens=self.n_tokens,
            embedding_dim=self.embedding_dim,
            similarity_threshold=self.similarity_threshold,
        )

    @classmethod
    def load(cls, filepath: str) -> 'EmbeddingTokenizer':
        """Load tokenizer from file."""
        data = np.load(filepath)

        tokenizer = cls(
            n_tokens=int(data['n_tokens']),
            embedding_dim=int(data['embedding_dim']),
            similarity_threshold=float(data['similarity_threshold']),
        )
        tokenizer.centroids = data['centroids']
        tokenizer.counts = data['counts']
        tokenizer._centroid_sums = data['centroid_sums']
        tokenizer.n_active = int(data['n_active'])

        return tokenizer

    def __repr__(self) -> str:
        return (f"EmbeddingTokenizer(n_tokens={self.n_tokens}, "
                f"n_active={self.n_active}, threshold={self.similarity_threshold})")


class HybridTokenizer:
    """
    Tokenizer that combines CLIP embedding + YOLO object histogram.

    Creates richer tokens that capture both visual appearance and semantic objects.
    """

    def __init__(
        self,
        n_tokens: int = 32,
        embedding_dim: int = 384,
        n_object_types: int = 17,
        embedding_weight: float = 0.7,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize hybrid tokenizer.

        Args:
            n_tokens: Maximum tokens
            embedding_dim: Image encoder dimension (384 for DINOv2, 512 for CLIP)
            n_object_types: Number of YOLO object types
            embedding_weight: Weight for encoder vs YOLO (0-1)
            similarity_threshold: Similarity threshold for assignment
        """
        self.embedding_weight = embedding_weight
        self.object_weight = 1.0 - embedding_weight
        self.n_object_types = n_object_types

        # Combined dimension: CLIP + YOLO histogram
        self.combined_dim = embedding_dim + n_object_types

        # Internal tokenizer on combined features
        self._tokenizer = EmbeddingTokenizer(
            n_tokens=n_tokens,
            embedding_dim=self.combined_dim,
            similarity_threshold=similarity_threshold,
        )

    def tokenize(
        self,
        embedding: np.ndarray,
        object_histogram: np.ndarray
    ) -> Tuple[int, float]:
        """
        Tokenize combined CLIP + YOLO features.

        Args:
            embedding: CLIP embedding (embedding_dim,)
            object_histogram: YOLO object counts/confidences (n_object_types,)

        Returns:
            token: Token index
            similarity: Similarity to assigned centroid
        """
        combined = self._combine_features(embedding, object_histogram)
        return self._tokenizer.tokenize(combined)

    def get_token(
        self,
        embedding: np.ndarray,
        object_histogram: np.ndarray
    ) -> Tuple[int, float]:
        """Get token without updating (inference only)."""
        combined = self._combine_features(embedding, object_histogram)
        return self._tokenizer.get_token(combined)

    def _combine_features(
        self,
        embedding: np.ndarray,
        object_histogram: np.ndarray
    ) -> np.ndarray:
        """Combine and weight CLIP + YOLO features."""
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        object_histogram = np.asarray(object_histogram, dtype=np.float32).flatten()

        # Normalize each
        emb_norm = np.linalg.norm(embedding)
        if emb_norm > 0:
            embedding = embedding / emb_norm

        obj_norm = np.linalg.norm(object_histogram)
        if obj_norm > 0:
            object_histogram = object_histogram / obj_norm

        # Weight and concatenate
        combined = np.concatenate([
            embedding * self.embedding_weight,
            object_histogram * self.object_weight
        ])

        return combined

    @property
    def n_active(self) -> int:
        return self._tokenizer.n_active

    @property
    def n_tokens(self) -> int:
        return self._tokenizer.n_tokens

    @property
    def counts(self) -> np.ndarray:
        return self._tokenizer.counts

    @property
    def centroids(self) -> np.ndarray:
        return self._tokenizer.centroids

    def save(self, filepath: str):
        """Save tokenizer."""
        self._tokenizer.save(filepath)

    @classmethod
    def load(cls, filepath: str, n_object_types: int = 17) -> 'HybridTokenizer':
        """Load tokenizer."""
        inner = EmbeddingTokenizer.load(filepath)

        # Reconstruct hybrid tokenizer
        embedding_dim = inner.embedding_dim - n_object_types
        tokenizer = cls(
            n_tokens=inner.n_tokens,
            embedding_dim=embedding_dim,
            n_object_types=n_object_types,
            similarity_threshold=inner.similarity_threshold,
        )
        tokenizer._tokenizer = inner
        return tokenizer

    def __repr__(self) -> str:
        return (f"HybridTokenizer(n_tokens={self.n_tokens}, "
                f"n_active={self.n_active}, "
                f"emb_weight={self.embedding_weight})")
