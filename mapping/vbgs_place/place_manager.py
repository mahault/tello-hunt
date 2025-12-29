"""
Place Manager for VBGS local place models.

Manages a collection of place models, one per observation token/clone.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from .place_model import VBGSPlaceModel, SimplePlaceModel
from .keyframe_selector import KeyframeSelector, AdaptiveKeyframeSelector


class PlaceManager:
    """
    Manages VBGS place models for all discovered places.

    Each place (identified by token or clone state) gets its own
    appearance model. The manager handles:
    - Creating new place models
    - Updating models with keyframes
    - Computing place evidence (ELBO/likelihood)
    - Persistence

    Attributes:
        places: Dict mapping place_id to place model
        keyframe_selector: Selector for when to add keyframes
        use_simple_model: Use SimplePlaceModel instead of full VBGS
    """

    def __init__(
        self,
        use_simple_model: bool = True,
        n_components: int = 64,
        embedding_dim: int = 512,
        min_keyframe_interval: float = 0.5,
        keyframe_threshold: float = 0.15,
    ):
        """
        Initialize place manager.

        Args:
            use_simple_model: Use simplified place model (recommended for real-time)
            n_components: Gaussian components per place (if using VBGS)
            embedding_dim: CLIP embedding dimension
            min_keyframe_interval: Minimum seconds between keyframes
            keyframe_threshold: Embedding distance threshold for keyframes
        """
        self.use_simple_model = use_simple_model
        self.n_components = n_components
        self.embedding_dim = embedding_dim

        # Place models: place_id -> model
        self.places: Dict[int, object] = {}

        # Keyframe selectors per place
        self._keyframe_selectors: Dict[int, KeyframeSelector] = {}

        # Global keyframe selector settings
        self._min_interval = min_keyframe_interval
        self._threshold = keyframe_threshold

    def _get_or_create_place(self, place_id: int) -> object:
        """Get existing place model or create new one."""
        if place_id not in self.places:
            if self.use_simple_model:
                self.places[place_id] = SimplePlaceModel(
                    embedding_dim=self.embedding_dim
                )
            else:
                self.places[place_id] = VBGSPlaceModel(
                    n_components=self.n_components
                )

            self._keyframe_selectors[place_id] = AdaptiveKeyframeSelector(
                min_interval=self._min_interval,
                base_threshold=self._threshold,
            )

        return self.places[place_id]

    def update_place(
        self,
        place_id: int,
        frame: np.ndarray = None,
        embedding: np.ndarray = None,
        force: bool = False,
    ) -> bool:
        """
        Update place model with new keyframe.

        Args:
            place_id: Place identifier (token or clone state)
            frame: BGR image frame (for VBGS model)
            embedding: CLIP embedding (for simple model)
            force: Force update even if keyframe criteria not met

        Returns:
            True if keyframe was added
        """
        model = self._get_or_create_place(place_id)
        selector = self._keyframe_selectors[place_id]

        # Check if should add keyframe
        if not force and embedding is not None:
            if not selector.should_add_keyframe(embedding, model.n_keyframes):
                return False

        # Update model
        if self.use_simple_model:
            if embedding is not None:
                model.update(embedding)
                if embedding is not None:
                    selector.mark_keyframe(embedding)
                return True
        else:
            if frame is not None:
                model.update(frame)
                if embedding is not None:
                    selector.mark_keyframe(embedding)
                return True

        return False

    def compute_place_evidence(
        self,
        frame: np.ndarray = None,
        embedding: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute evidence (likelihood) for all places.

        Args:
            frame: BGR image frame (for VBGS model)
            embedding: CLIP embedding (for simple model)

        Returns:
            evidence: Array of likelihoods, one per place
        """
        if len(self.places) == 0:
            return np.array([])

        max_id = max(self.places.keys()) + 1
        evidence = np.zeros(max_id, dtype=np.float32)

        for place_id, model in self.places.items():
            if self.use_simple_model:
                if embedding is not None:
                    evidence[place_id] = model.compute_likelihood(embedding)
            else:
                if frame is not None:
                    evidence[place_id] = model.compute_elbo(frame)

        return evidence

    def compute_place_similarities(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities to all place models.

        Args:
            embedding: Query embedding

        Returns:
            similarities: Array of similarities, one per place
        """
        if len(self.places) == 0:
            return np.array([])

        max_id = max(self.places.keys()) + 1
        similarities = np.zeros(max_id, dtype=np.float32)

        for place_id, model in self.places.items():
            if hasattr(model, 'compute_similarity'):
                similarities[place_id] = model.compute_similarity(embedding)

        return similarities

    def get_best_place(
        self,
        frame: np.ndarray = None,
        embedding: np.ndarray = None,
    ) -> Tuple[int, float]:
        """
        Get the place that best matches the current observation.

        Args:
            frame: BGR image frame
            embedding: CLIP embedding

        Returns:
            (place_id, score) of best matching place
        """
        if self.use_simple_model and embedding is not None:
            similarities = self.compute_place_similarities(embedding)
            if len(similarities) == 0:
                return -1, 0.0
            best_id = int(np.argmax(similarities))
            return best_id, float(similarities[best_id])
        else:
            evidence = self.compute_place_evidence(frame, embedding)
            if len(evidence) == 0:
                return -1, 0.0
            best_id = int(np.argmax(evidence))
            return best_id, float(evidence[best_id])

    def get_place_info(self, place_id: int) -> Dict:
        """Get information about a place model."""
        if place_id not in self.places:
            return {'error': 'Place not found'}

        model = self.places[place_id]
        selector = self._keyframe_selectors.get(place_id)

        return {
            'place_id': place_id,
            'n_keyframes': model.n_keyframes,
            'model_type': type(model).__name__,
            'selector_count': selector.keyframe_count if selector else 0,
        }

    @property
    def n_places(self) -> int:
        """Number of place models."""
        return len(self.places)

    @property
    def total_keyframes(self) -> int:
        """Total keyframes across all places."""
        return sum(m.n_keyframes for m in self.places.values())

    def save(self, directory: str):
        """Save all place models."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        np.savez(
            str(path / 'config.npz'),
            use_simple_model=self.use_simple_model,
            n_components=self.n_components,
            embedding_dim=self.embedding_dim,
            min_interval=self._min_interval,
            threshold=self._threshold,
        )

        # Save place IDs
        place_ids = list(self.places.keys())
        np.save(str(path / 'place_ids.npy'), place_ids)

        # Save each place model
        for place_id, model in self.places.items():
            model.save(str(path / f'place_{place_id}.npz'))

    @classmethod
    def load(cls, directory: str) -> 'PlaceManager':
        """Load place manager from directory."""
        path = Path(directory)

        # Load config
        config = dict(np.load(str(path / 'config.npz')))

        manager = cls(
            use_simple_model=bool(config['use_simple_model']),
            n_components=int(config['n_components']),
            embedding_dim=int(config['embedding_dim']),
            min_keyframe_interval=float(config['min_interval']),
            keyframe_threshold=float(config['threshold']),
        )

        # Load place IDs
        place_ids = np.load(str(path / 'place_ids.npy'))

        # Load each place model
        for place_id in place_ids:
            model_path = path / f'place_{place_id}.npz'
            if model_path.exists():
                if manager.use_simple_model:
                    manager.places[int(place_id)] = SimplePlaceModel.load(str(model_path))
                else:
                    manager.places[int(place_id)] = VBGSPlaceModel.load(str(model_path))

                manager._keyframe_selectors[int(place_id)] = AdaptiveKeyframeSelector(
                    min_interval=manager._min_interval,
                    base_threshold=manager._threshold,
                )

        return manager

    def __repr__(self) -> str:
        return (f"PlaceManager(n_places={self.n_places}, "
                f"total_keyframes={self.total_keyframes}, "
                f"simple={self.use_simple_model})")
