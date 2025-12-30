"""
Monocular Depth Estimation for Tello Drone.

Uses neural networks to estimate depth from single RGB images,
which is what the actual Tello drone would need to do (no depth sensor).

Supports:
- Depth Anything v2 (recommended, best quality)
- MiDaS (fallback, well-established)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path


class DepthEstimator:
    """
    Monocular depth estimation from RGB images.

    Estimates relative depth (not metric) from single frames.
    For metric depth, would need camera calibration + scale factor.

    Uses robust normalization (percentiles) and temporal smoothing
    for stable collision detection.
    """

    def __init__(self, model_type: str = "depth_anything", device: str = "auto"):
        """
        Initialize depth estimator.

        Args:
            model_type: "depth_anything" or "midas"
            device: "auto", "cuda", or "cpu"
        """
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.device = None

        # Temporal smoothing for stable collision detection
        self._collision_ema: Dict[str, float] = {}  # EMA per region
        self._ema_alpha = 0.3  # Smoothing factor (0=slow, 1=instant)
        self._blocked_count: Dict[str, int] = {}  # Debounce counters
        self._blocked_threshold = 3  # Frames to confirm blocked

        # Try to load the requested model
        self._load_model(model_type, device)

    def _load_model(self, model_type: str, device: str):
        """Load the depth estimation model."""
        import torch

        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading depth estimator ({model_type}) on {self.device}...")

        if model_type == "depth_anything":
            try:
                self._load_depth_anything()
                return
            except Exception as e:
                print(f"Depth Anything failed: {e}")
                print("Falling back to MiDaS...")
                model_type = "midas"

        if model_type == "midas":
            self._load_midas()

    def _load_depth_anything(self):
        """Load Depth Anything v2 model."""
        import torch
        from transformers import pipeline

        # Use the small model for speed (vits), medium (vitb), or large (vitl)
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if self.device.type == "cuda" else -1,
        )
        self.model_type = "depth_anything"
        print("Depth Anything v2 loaded!")

    def _load_midas(self):
        """Load MiDaS model from torch hub."""
        import torch

        # MiDaS small is fast, large is more accurate
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

        self.model_type = "midas"
        print("MiDaS loaded!")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth from RGB frame.

        Args:
            frame: BGR image (H, W, 3) from OpenCV

        Returns:
            Depth map (H, W) with values 0-1 (0=far, 1=near)
            Normalized for display/comparison, not metric.
        """
        if self.model_type == "depth_anything":
            return self._estimate_depth_anything(frame)
        else:
            return self._estimate_midas(frame)

    def _estimate_depth_anything(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using Depth Anything."""
        from PIL import Image

        # Convert BGR to RGB PIL image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Run inference
        result = self.pipe(pil_image)
        depth = np.array(result["depth"])

        # Robust normalization using percentiles (not min/max)
        # This prevents single outlier pixels from rescaling the whole map
        depth = depth.astype(np.float32)
        p2, p98 = np.percentile(depth, [2, 98])
        depth = np.clip((depth - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Resize to match input frame
        if depth.shape[:2] != frame.shape[:2]:
            depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

        return depth

    def _estimate_midas(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using MiDaS."""
        import torch

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transforms
        input_batch = self.transform(rgb).to(self.device)

        # Inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # Robust normalization using percentiles (not min/max)
        # MiDaS outputs inverse depth, so near=high
        p2, p98 = np.percentile(depth, [2, 98])
        depth = np.clip((depth - p2) / (p98 - p2 + 1e-8), 0, 1)

        return depth

    def get_obstacle_distances(
        self,
        depth: np.ndarray,
        regions: Dict[str, Tuple[float, float, float, float]] = None,
    ) -> Dict[str, float]:
        """
        Get relative distances to obstacles in different regions.

        Args:
            depth: Depth map from estimate()
            regions: Dict of region_name -> (x1, y1, x2, y2) as fractions 0-1
                    Default regions: center, left, right, top, bottom

        Returns:
            Dict of region_name -> relative distance (0=blocked, 1=clear)
        """
        h, w = depth.shape

        if regions is None:
            # Default regions (as fractions of image)
            regions = {
                'center': (0.35, 0.35, 0.65, 0.65),  # Center square
                'left': (0.0, 0.3, 0.25, 0.7),       # Left side
                'right': (0.75, 0.3, 1.0, 0.7),      # Right side
                'top': (0.3, 0.0, 0.7, 0.25),        # Top
                'bottom': (0.3, 0.75, 0.7, 1.0),     # Bottom
                'forward_path': (0.3, 0.4, 0.7, 0.8), # Forward flight path
            }

        distances = {}
        for name, (x1, y1, x2, y2) in regions.items():
            # Convert fractions to pixels
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)

            # Get region
            region = depth[py1:py2, px1:px2]

            if region.size == 0:
                distances[name] = 0.0
                continue

            # Use median of closest 25% of pixels (obstacles)
            sorted_depths = np.sort(region.flatten())
            closest_quarter = sorted_depths[:len(sorted_depths)//4 + 1]

            # Higher depth value = closer (inverted)
            # Return 1 - median so that 1=far/clear, 0=near/blocked
            distances[name] = 1.0 - float(np.median(closest_quarter))

        return distances

    def is_path_clear(
        self,
        depth: np.ndarray,
        direction: str = "forward",
        threshold: float = 0.3,
    ) -> Tuple[bool, float]:
        """
        Check if path is clear in given direction.

        Uses temporal EMA smoothing and debouncing for stable detection.

        Args:
            depth: Depth map from estimate()
            direction: "forward", "left", "right", "up", "down"
            threshold: Minimum distance (0-1) to consider clear

        Returns:
            (is_clear, smoothed_distance) tuple
        """
        distances = self.get_obstacle_distances(depth)

        direction_map = {
            'forward': 'forward_path',
            'left': 'left',
            'right': 'right',
            'up': 'top',
            'down': 'bottom',
        }

        region = direction_map.get(direction, 'center')
        raw_distance = distances.get(region, 0.0)

        # Apply temporal EMA smoothing
        if region not in self._collision_ema:
            self._collision_ema[region] = raw_distance
        else:
            self._collision_ema[region] = (
                self._ema_alpha * raw_distance +
                (1 - self._ema_alpha) * self._collision_ema[region]
            )

        smoothed_distance = self._collision_ema[region]

        # Debounce: require K consecutive frames to confirm blocked
        if region not in self._blocked_count:
            self._blocked_count[region] = 0

        if smoothed_distance < threshold:
            self._blocked_count[region] += 1
        else:
            self._blocked_count[region] = max(0, self._blocked_count[region] - 1)

        # Only report blocked if debounce threshold met
        is_blocked = self._blocked_count[region] >= self._blocked_threshold
        is_clear = not is_blocked

        return is_clear, smoothed_distance

    def visualize_depth(
        self,
        depth: np.ndarray,
        colormap: int = cv2.COLORMAP_MAGMA,
    ) -> np.ndarray:
        """
        Create colorized depth visualization.

        Args:
            depth: Depth map (0-1)
            colormap: OpenCV colormap

        Returns:
            BGR image for display
        """
        # Convert to uint8
        depth_uint8 = (depth * 255).astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(depth_uint8, colormap)

        return colored

    def visualize_with_obstacles(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
        distances: Dict[str, float] = None,
    ) -> np.ndarray:
        """
        Create visualization with obstacle warnings overlaid.

        Args:
            frame: Original BGR frame
            depth: Depth map
            distances: Optional pre-computed distances

        Returns:
            BGR image with depth overlay and warnings
        """
        if distances is None:
            distances = self.get_obstacle_distances(depth)

        # Create depth colormap
        depth_colored = self.visualize_depth(depth)

        # Blend with original frame
        blended = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)

        h, w = frame.shape[:2]

        # Draw distance indicators
        regions_coords = {
            'center': (w//2, h//2),
            'left': (w//6, h//2),
            'right': (5*w//6, h//2),
            'forward_path': (w//2, 2*h//3),
        }

        for name, (cx, cy) in regions_coords.items():
            if name not in distances:
                continue

            dist = distances[name]

            # Color based on distance (red=close, green=far)
            if dist < 0.3:
                color = (0, 0, 255)  # Red - danger
            elif dist < 0.5:
                color = (0, 165, 255)  # Orange - caution
            else:
                color = (0, 255, 0)  # Green - clear

            # Draw circle and distance
            cv2.circle(blended, (cx, cy), 20, color, 2)
            cv2.putText(blended, f"{dist:.2f}", (cx-20, cy+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Warning if forward path blocked
        if distances.get('forward_path', 1.0) < 0.3:
            cv2.putText(blended, "OBSTACLE AHEAD!", (w//2 - 80, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return blended


def test_depth_estimator():
    """Test depth estimation on webcam or sample images."""
    print("Testing Depth Estimator...")

    estimator = DepthEstimator(model_type="depth_anything")

    # Try webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam available, using test pattern")
        # Create test image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 150), (440, 330), (255, 255, 255), -1)
        frames = [frame]
    else:
        frames = None

    print("Press 'q' to quit")

    while True:
        if frames:
            frame = frames[0]
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # Estimate depth
        depth = estimator.estimate(frame)

        # Get obstacle distances
        distances = estimator.get_obstacle_distances(depth)

        # Visualize
        vis = estimator.visualize_with_obstacles(frame, depth, distances)

        # Show
        cv2.imshow("Depth Estimation", vis)

        key = cv2.waitKey(1 if frames is None else 0) & 0xFF
        if key == ord('q'):
            break

    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_depth_estimator()
