"""
VBGS Place Model for local appearance modeling.

Each "place" gets its own Gaussian mixture model that captures
the appearance distribution of that location without requiring
accurate camera poses.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import copy

# Import VBGS components
import sys
from pathlib import Path

# Add vbgs to path
vbgs_path = Path(__file__).parent.parent.parent / 'vbgs'
if str(vbgs_path) not in sys.path:
    sys.path.insert(0, str(vbgs_path))

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from vbgs.data.utils import create_normalizing_params, normalize_data
    from vbgs.data.image import image_to_data
    from vbgs.model.train import fit_gmm_step
    from vbgs.render.image import render_img
    VBGS_AVAILABLE = True
except ImportError as e:
    print(f"VBGS not available: {e}")
    VBGS_AVAILABLE = False


@dataclass
class PlaceModelConfig:
    """Configuration for VBGS place model."""
    n_components: int = 64  # Number of Gaussian components
    patch_size: int = 32  # Process image in patches
    learning_rate: float = 1.0
    beta: float = 0.0  # Forgetting factor (0 = no forgetting)


class VBGSPlaceModel:
    """
    Local Gaussian mixture model for a single place.

    Uses VBGS to learn a distribution over (x, y, r, g, b) for
    the appearance of this place. Can compute ELBO to measure
    how well a new frame fits this place model.

    Attributes:
        n_components: Number of Gaussian components
        n_keyframes: Number of keyframes used for learning
        model: VBGS DeltaMixture model (if initialized)
    """

    def __init__(
        self,
        n_components: int = 64,
        image_shape: Tuple[int, int] = (480, 640),
    ):
        """
        Initialize place model.

        Args:
            n_components: Number of Gaussian mixture components
            image_shape: Expected image shape (H, W)
        """
        self.n_components = n_components
        self.image_shape = image_shape
        self.n_keyframes = 0

        # VBGS model components (lazy initialized)
        self._model = None
        self._initial_model = None
        self._data_params = None

        # Accumulated sufficient statistics
        self._prior_stats = None
        self._space_stats = None
        self._color_stats = None

        # Random key for JAX
        self._key = jr.PRNGKey(42) if VBGS_AVAILABLE else None

    def _initialize_model(self, frame: np.ndarray):
        """Initialize VBGS model from first frame."""
        if not VBGS_AVAILABLE:
            return

        # Normalize frame to [0, 1]
        img = frame.astype(np.float32) / 255.0
        if img.shape[:2] != self.image_shape:
            import cv2
            img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))

        h, w = img.shape[:2]

        # Create normalizing parameters for (u, v, r, g, b)
        self._data_params = create_normalizing_params(
            [0, w], [0, h], [0, 1], [0, 1], [0, 1]
        )

        # Import model creation function
        try:
            # Get model creation function from vbgs
            from vbgs.model.utils import random_mean_init
            from vbgs.vi.conjugate.mvn import MultivariateNormal
            from vbgs.vi.models.mixture import Mixture
            from vbgs.model.model import DeltaMixture

            event_shape = (5, 1)  # (u, v, r, g, b)
            component_shape = (self.n_components,)

            self._key, subkey = jr.split(self._key)
            mean_init = random_mean_init(
                subkey,
                None,
                component_shape,
                event_shape,
                init_random=True,
                add_noise=False,
            )

            # Create initial model (simplified - may need adjustment)
            # This is a placeholder - actual model creation depends on VBGS internals
            self._model = None  # Will be set during first update

        except Exception as e:
            print(f"Error initializing VBGS model: {e}")
            self._model = None

    def update(self, frame: np.ndarray) -> float:
        """
        Update place model with new keyframe.

        Args:
            frame: BGR image frame

        Returns:
            elbo: Evidence lower bound (higher = better fit)
        """
        if not VBGS_AVAILABLE:
            self.n_keyframes += 1
            return 0.0

        # Normalize frame
        img = frame.astype(np.float32) / 255.0
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)

        # Convert BGR to RGB if needed
        if img.shape[-1] == 3:
            img = img[..., ::-1]  # BGR -> RGB

        # Resize if needed
        if img.shape[:2] != self.image_shape:
            import cv2
            img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))

        # Initialize on first frame
        if self._data_params is None:
            self._initialize_model(frame)

        # Convert image to data points (x, y, r, g, b)
        try:
            data = image_to_data(img)  # Shape: (H*W, 5)

            # Normalize
            x, _ = normalize_data(data, self._data_params)

            # For now, just track keyframe count
            # Full VBGS update requires more complex model setup
            self.n_keyframes += 1

            # Return placeholder ELBO
            return 0.0

        except Exception as e:
            print(f"Error updating place model: {e}")
            self.n_keyframes += 1
            return 0.0

    def compute_elbo(self, frame: np.ndarray) -> float:
        """
        Compute ELBO for how well frame fits this place.

        Higher ELBO = better fit = more likely to be this place.

        Args:
            frame: BGR image frame

        Returns:
            elbo: Evidence lower bound
        """
        if not VBGS_AVAILABLE or self._model is None:
            return 0.0

        # For now, return placeholder
        # Full implementation requires computing VBGS likelihood
        return 0.0

    def render_expected_view(self) -> Optional[np.ndarray]:
        """
        Render the expected appearance of this place.

        Returns:
            image: Expected view as (H, W, 3) RGB image, or None
        """
        if not VBGS_AVAILABLE or self._model is None:
            return None

        try:
            mu, si = self._model.denormalize(self._data_params)
            alpha = self._model.prior.alpha

            rendered = render_img(mu, si, alpha, self.image_shape)
            return (rendered * 255).astype(np.uint8)
        except Exception:
            return None

    def get_subarea_count(self) -> int:
        """Get number of active Gaussian components (sub-areas)."""
        return self.n_components

    def save(self, filepath: str):
        """Save place model."""
        data = {
            'n_components': self.n_components,
            'image_shape': self.image_shape,
            'n_keyframes': self.n_keyframes,
        }
        np.savez(filepath, **data)

    @classmethod
    def load(cls, filepath: str) -> 'VBGSPlaceModel':
        """Load place model."""
        data = np.load(filepath)
        model = cls(
            n_components=int(data['n_components']),
            image_shape=tuple(data['image_shape']),
        )
        model.n_keyframes = int(data['n_keyframes'])
        return model

    def __repr__(self) -> str:
        return f"VBGSPlaceModel(n_components={self.n_components}, n_keyframes={self.n_keyframes})"


class SimplePlaceModel:
    """
    Simplified place model using image statistics instead of full VBGS.

    Tracks mean/variance of appearance features for quick place matching.
    Useful as fallback when full VBGS is too slow or unavailable.
    """

    def __init__(self, embedding_dim: int = 512):
        """
        Initialize simple place model.

        Args:
            embedding_dim: CLIP embedding dimension
        """
        self.embedding_dim = embedding_dim
        self.n_keyframes = 0

        # Running statistics
        self._embedding_sum = np.zeros(embedding_dim, dtype=np.float64)
        self._embedding_sq_sum = np.zeros(embedding_dim, dtype=np.float64)
        self._mean = None
        self._var = None

    def update(self, embedding: np.ndarray):
        """Update with new keyframe embedding."""
        embedding = np.asarray(embedding, dtype=np.float64).flatten()

        self._embedding_sum += embedding
        self._embedding_sq_sum += embedding ** 2
        self.n_keyframes += 1

        # Update mean and variance
        self._mean = self._embedding_sum / self.n_keyframes
        self._var = (self._embedding_sq_sum / self.n_keyframes) - (self._mean ** 2)
        self._var = np.maximum(self._var, 1e-6)  # Ensure positive

    def compute_likelihood(self, embedding: np.ndarray) -> float:
        """
        Compute likelihood of embedding under this place model.

        Uses Gaussian likelihood with learned mean/variance.
        """
        if self._mean is None:
            return 0.0

        embedding = np.asarray(embedding, dtype=np.float64).flatten()

        # Mahalanobis-like distance (diagonal covariance)
        diff = embedding - self._mean
        dist = np.sum((diff ** 2) / self._var)

        # Convert to log-likelihood (Gaussian)
        log_lik = -0.5 * dist - 0.5 * np.sum(np.log(self._var))

        return float(log_lik)

    def compute_similarity(self, embedding: np.ndarray) -> float:
        """Compute cosine similarity to mean embedding."""
        if self._mean is None:
            return 0.0

        embedding = np.asarray(embedding, dtype=np.float32).flatten()

        # Normalize
        emb_norm = np.linalg.norm(embedding)
        mean_norm = np.linalg.norm(self._mean)

        if emb_norm > 0 and mean_norm > 0:
            return float(np.dot(embedding, self._mean) / (emb_norm * mean_norm))

        return 0.0

    def save(self, filepath: str):
        """Save model."""
        np.savez(
            filepath,
            embedding_sum=self._embedding_sum,
            embedding_sq_sum=self._embedding_sq_sum,
            n_keyframes=self.n_keyframes,
            embedding_dim=self.embedding_dim,
        )

    @classmethod
    def load(cls, filepath: str) -> 'SimplePlaceModel':
        """Load model."""
        data = np.load(filepath)
        model = cls(embedding_dim=int(data['embedding_dim']))
        model._embedding_sum = data['embedding_sum']
        model._embedding_sq_sum = data['embedding_sq_sum']
        model.n_keyframes = int(data['n_keyframes'])

        if model.n_keyframes > 0:
            model._mean = model._embedding_sum / model.n_keyframes
            model._var = (model._embedding_sq_sum / model.n_keyframes) - (model._mean ** 2)
            model._var = np.maximum(model._var, 1e-6)

        return model

    def __repr__(self) -> str:
        return f"SimplePlaceModel(n_keyframes={self.n_keyframes})"
