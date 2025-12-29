"""
Tentative spatial map from action-based dead reckoning.

Builds a 2D layout of tokens based on movement actions:
- Forward (action 1) = +Y (up on screen)
- Back (action 2) = -Y (down)
- Left/rotate left (action 3) = updates heading
- Right/rotate right (action 4) = updates heading

This is NOT accurate metric mapping - just a rough visualization
of how places connect spatially based on actions taken.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
import math


class SpatialMap:
    """
    Builds tentative spatial positions from movement actions.
    """

    def __init__(self, width: int = 400, height: int = 400):
        self.width = width
        self.height = height

        # Token positions in world coordinates (arbitrary units)
        self._positions: Dict[int, Tuple[float, float]] = {}

        # Current estimated pose (x, y, heading in radians)
        self._x = 0.0
        self._y = 0.0
        self._heading = math.pi / 2  # Start facing "up" (north)

        # Movement scale
        self._step_size = 1.0
        self._turn_angle = math.pi / 4  # 45 degrees per turn action

        # Tracking
        self._prev_token = -1
        self._current_token = -1

        # For rendering
        self._cached_render: Optional[np.ndarray] = None
        self._render_dirty = True

    def update(self, token: int, action: int):
        """
        Update map with new observation and action taken.

        Args:
            token: Current observation token
            action: Action that was taken (0=stay, 1=fwd, 2=back, 3=left, 4=right)
        """
        # Update pose based on action
        if action == 1:  # Forward
            self._x += self._step_size * math.cos(self._heading)
            self._y += self._step_size * math.sin(self._heading)
        elif action == 2:  # Back
            self._x -= self._step_size * math.cos(self._heading)
            self._y -= self._step_size * math.sin(self._heading)
        elif action == 3:  # Left (rotate)
            self._heading += self._turn_angle
        elif action == 4:  # Right (rotate)
            self._heading -= self._turn_angle

        # Normalize heading
        self._heading = self._heading % (2 * math.pi)

        # If new token, record its position
        if token not in self._positions:
            self._positions[token] = (self._x, self._y)
            self._render_dirty = True
        elif token != self._prev_token:
            # Revisiting a token - average position for stability
            old_x, old_y = self._positions[token]
            self._positions[token] = (
                0.7 * old_x + 0.3 * self._x,
                0.7 * old_y + 0.3 * self._y
            )
            # Snap our position to known location
            self._x, self._y = self._positions[token]

        self._prev_token = self._current_token
        self._current_token = token

        if token != self._prev_token:
            self._render_dirty = True

    def render(self) -> np.ndarray:
        """
        Render the spatial map.

        Returns:
            BGR image showing room layout
        """
        canvas = np.full((self.height, self.width, 3), (40, 35, 30), dtype=np.uint8)

        if len(self._positions) == 0:
            cv2.putText(canvas, "Exploring...", (self.width//2 - 50, self.height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            return canvas

        # Compute bounds
        xs = [p[0] for p in self._positions.values()]
        ys = [p[1] for p in self._positions.values()]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Add current position to bounds
        min_x = min(min_x, self._x)
        max_x = max(max_x, self._x)
        min_y = min(min_y, self._y)
        max_y = max(max_y, self._y)

        # Compute scale to fit in canvas with margin
        margin = 60
        range_x = max(max_x - min_x, 0.1)
        range_y = max(max_y - min_y, 0.1)

        scale = min(
            (self.width - 2 * margin) / range_x,
            (self.height - 2 * margin) / range_y
        )
        scale = min(scale, 80)  # Cap scale for very small maps

        # Center offset
        cx = self.width // 2
        cy = self.height // 2
        offset_x = (min_x + max_x) / 2
        offset_y = (min_y + max_y) / 2

        def world_to_screen(wx: float, wy: float) -> Tuple[int, int]:
            sx = int(cx + (wx - offset_x) * scale)
            sy = int(cy - (wy - offset_y) * scale)  # Flip Y for screen coords
            return (sx, sy)

        # Draw connections between adjacent tokens (based on transitions)
        # For now, just draw lines between tokens that are close
        token_list = list(self._positions.keys())
        for i, t1 in enumerate(token_list):
            p1 = self._positions[t1]
            for t2 in token_list[i+1:]:
                p2 = self._positions[t2]
                dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                if dist < self._step_size * 1.5:
                    s1 = world_to_screen(*p1)
                    s2 = world_to_screen(*p2)
                    cv2.line(canvas, s1, s2, (80, 70, 60), 2)

        # Draw tokens as rooms
        for token, (wx, wy) in self._positions.items():
            sx, sy = world_to_screen(wx, wy)

            # Room box
            room_size = 25
            color = (60, 100, 60)

            # Highlight current
            if token == self._current_token:
                color = (80, 180, 80)
                cv2.rectangle(canvas,
                            (sx - room_size - 3, sy - room_size - 3),
                            (sx + room_size + 3, sy + room_size + 3),
                            (0, 255, 255), 2)

            cv2.rectangle(canvas,
                        (sx - room_size, sy - room_size),
                        (sx + room_size, sy + room_size),
                        color, -1)
            cv2.rectangle(canvas,
                        (sx - room_size, sy - room_size),
                        (sx + room_size, sy + room_size),
                        (100, 100, 100), 1)

            # Token label
            cv2.putText(canvas, f"T{token}", (sx - 10, sy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Draw drone position and heading
        dx, dy = world_to_screen(self._x, self._y)

        # Heading arrow
        arrow_len = 20
        ax = int(dx + arrow_len * math.cos(-self._heading + math.pi/2))
        ay = int(dy + arrow_len * math.sin(-self._heading + math.pi/2))
        cv2.arrowedLine(canvas, (dx, dy), (ax, ay), (0, 200, 255), 2, tipLength=0.4)

        # Drone marker
        cv2.circle(canvas, (dx, dy), 6, (0, 200, 255), -1)

        # Title
        cv2.putText(canvas, "Spatial Map (estimated)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"Places: {len(self._positions)}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Compass
        self._draw_compass(canvas, self.width - 40, 40)

        return canvas

    def _draw_compass(self, canvas: np.ndarray, cx: int, cy: int):
        """Draw a small compass."""
        r = 20
        cv2.circle(canvas, (cx, cy), r, (80, 80, 80), 1)
        cv2.line(canvas, (cx, cy - r + 5), (cx, cy - r + 12), (150, 150, 200), 2)
        cv2.putText(canvas, "N", (cx - 4, cy - r - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 200), 1)

    def reset(self):
        """Reset the map."""
        self._positions.clear()
        self._x = 0.0
        self._y = 0.0
        self._heading = math.pi / 2
        self._prev_token = -1
        self._current_token = -1
        self._render_dirty = True


def combine_with_map(
    camera_frame: np.ndarray,
    spatial_map: np.ndarray,
    target_height: int = 480
) -> np.ndarray:
    """
    Combine camera frame with spatial map side-by-side.

    Args:
        camera_frame: Camera image
        spatial_map: Spatial map image
        target_height: Target height for output

    Returns:
        Combined image
    """
    # Resize both to target height
    ch, cw = camera_frame.shape[:2]
    mh, mw = spatial_map.shape[:2]

    cam_scale = target_height / ch
    map_scale = target_height / mh

    cam_resized = cv2.resize(camera_frame, (int(cw * cam_scale), target_height))
    map_resized = cv2.resize(spatial_map, (int(mw * map_scale), target_height))

    return np.hstack([cam_resized, map_resized])
