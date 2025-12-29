"""
Image embedding encoder for stable location signatures.

Uses CLIP (Contrastive Language-Image Pre-training) to extract
semantic image embeddings that are more stable than YOLO object
detection histograms for place recognition.

CLIP advantages:
- Trained on 400M image-text pairs
- Understands scene semantics
- Robust to viewpoint changes
- Captures "place-ness" better than object counts
"""

import numpy as np
from typing import Optional, Tuple
import cv2

# Lazy imports to avoid loading heavy models at module import
_clip_model = None
_clip_processor = None
_device = None


def _load_clip():
    """Lazy load CLIP model on first use."""
    global _clip_model, _clip_processor, _device

    if _clip_model is not None:
        return _clip_model, _clip_processor, _device

    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel

        # Use GPU if available
        if torch.cuda.is_available():
            _device = "cuda"
            print(f"Loading CLIP model on GPU ({torch.cuda.get_device_name(0)})...")
        else:
            _device = "cpu"
            print("Loading CLIP model on CPU...")

        # Try offline first (for Tello WiFi), fall back to download
        try:
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        except Exception:
            print("Model not cached, downloading from HuggingFace...")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.to(_device)
        _clip_model.eval()
        print("CLIP model loaded.")

        return _clip_model, _clip_processor, _device

    except ImportError as e:
        raise ImportError(
            "CLIP requires transformers and torch. Install with:\n"
            "  pip install transformers torch"
        ) from e


# =============================================================================
# Embedding dimension
# =============================================================================

CLIP_EMBEDDING_DIM = 512  # CLIP ViT-B/32 output dimension


# =============================================================================
# Image Encoder Class
# =============================================================================

class ImageEncoder:
    """
    Encodes images to semantic embedding vectors using CLIP.

    Usage:
        encoder = ImageEncoder()
        embedding = encoder.encode(frame)  # Returns (512,) numpy array
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the image encoder.

        Args:
            model_name: HuggingFace model name for CLIP
        """
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None

        # Cache for temporal smoothing
        self._embedding_history: list = []
        self._max_history: int = 5

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._model is None:
            self._model, self._processor, self._device = _load_clip()

    def encode(
        self,
        frame: np.ndarray,
        normalize: bool = True,
        temporal_smooth: bool = True,
    ) -> np.ndarray:
        """
        Encode a frame to an embedding vector.

        Args:
            frame: BGR image from OpenCV (H, W, 3)
            normalize: Whether to L2-normalize the embedding
            temporal_smooth: Average with recent embeddings for stability

        Returns:
            Embedding vector of shape (512,)
        """
        self._ensure_loaded()

        import torch

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess for CLIP
        inputs = self._processor(images=rgb_frame, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Extract image features
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)

        # Convert to numpy
        embedding = image_features.cpu().numpy().squeeze()

        # L2 normalize
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        # Temporal smoothing
        if temporal_smooth:
            self._embedding_history.append(embedding)
            if len(self._embedding_history) > self._max_history:
                self._embedding_history.pop(0)

            # Average recent embeddings
            if len(self._embedding_history) > 1:
                embedding = np.mean(self._embedding_history, axis=0)
                # Re-normalize after averaging
                if normalize:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

        return embedding

    def encode_batch(self, frames: list) -> np.ndarray:
        """
        Encode multiple frames in a batch.

        Args:
            frames: List of BGR images

        Returns:
            Embeddings of shape (N, 512)
        """
        self._ensure_loaded()

        import torch

        # Convert all frames to RGB
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

        # Preprocess batch
        inputs = self._processor(images=rgb_frames, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)

        # Convert and normalize
        embeddings = image_features.cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        embeddings = embeddings / norms

        return embeddings

    def reset_history(self):
        """Clear temporal smoothing history."""
        self._embedding_history.clear()

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return CLIP_EMBEDDING_DIM


# =============================================================================
# Similarity Functions
# =============================================================================

def embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        emb1: First embedding (512,)
        emb2: Second embedding (512,)

    Returns:
        Similarity in range [-1, 1], higher = more similar
    """
    return float(np.dot(emb1, emb2))


def find_most_similar_embedding(
    query: np.ndarray,
    embeddings: np.ndarray,
) -> Tuple[int, float]:
    """
    Find the most similar embedding in a collection.

    Args:
        query: Query embedding (512,)
        embeddings: Collection of embeddings (N, 512)

    Returns:
        (best_index, best_similarity)
    """
    if len(embeddings) == 0:
        return -1, 0.0

    # All embeddings should be normalized, so dot product = cosine similarity
    similarities = np.dot(embeddings, query)
    best_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_idx])

    return best_idx, best_sim


# =============================================================================
# Singleton encoder for convenience
# =============================================================================

_encoder_instance: Optional[ImageEncoder] = None


def get_encoder() -> ImageEncoder:
    """Get or create the singleton image encoder."""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = ImageEncoder()
    return _encoder_instance


def encode_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convenience function to encode a frame.

    Args:
        frame: BGR image from OpenCV

    Returns:
        Embedding vector (512,)
    """
    return get_encoder().encode(frame)
