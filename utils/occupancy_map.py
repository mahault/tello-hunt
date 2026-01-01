"""
Occupancy Grid Mapping for Tello Drone.

Builds a 2D top-down map of the environment using:
- Drone pose from visual/action odometry
- Depth estimates from monocular depth estimation

The map shows:
- Free space (where drone can fly)
- Occupied space (obstacles)
- Unknown space (not yet observed)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class MapConfig:
    """Configuration for occupancy map."""
    width: int = 200           # Map width in cells
    height: int = 200          # Map height in cells
    resolution: float = 0.1    # Meters per cell
    origin_x: int = 100        # Origin X in cells (center)
    origin_y: int = 100        # Origin Y in cells (center)

    # Occupancy values
    unknown: int = 127         # Unknown space
    free: int = 255            # Free space
    occupied: int = 0          # Obstacle (collision-confirmed)
    depth_obstacle: int = 60   # Depth-sensed obstacle (low confidence)

    # Update parameters
    free_increment: int = 10   # How much to increase free confidence
    occupied_increment: int = 30  # How much to increase occupied confidence (collision)
    depth_occupied_increment: int = 12  # Weaker for depth-based obstacles


class OccupancyMap:
    """
    2D occupancy grid map.

    Accumulates observations to build a map of the environment.
    Uses log-odds for probabilistic updates.
    """

    def __init__(self, config: MapConfig = None):
        """Initialize occupancy map."""
        self.config = config or MapConfig()

        # Initialize map to unknown
        self.grid = np.full(
            (self.config.height, self.config.width),
            self.config.unknown,
            dtype=np.uint8
        )

        # Visit count for each cell
        self.visit_count = np.zeros(
            (self.config.height, self.config.width),
            dtype=np.int32
        )

        # Trajectory overlay
        self.trajectory: List[Tuple[int, int]] = []

        # Place markers: place_id -> (cx, cy)
        self.places: Dict[int, Tuple[int, int]] = {}

        # Semantic labels for places: place_id -> label string
        self.place_labels: Dict[int, str] = {}

    def world_to_map(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates (meters) to map coordinates (cells).

        Args:
            x: World X (forward from start)
            y: World Y (right from start)

        Returns:
            (map_x, map_y) cell coordinates
        """
        # Convert meters to cells
        cell_x = int(x / self.config.resolution) + self.config.origin_x
        cell_y = int(y / self.config.resolution) + self.config.origin_y

        return cell_x, cell_y

    def map_to_world(self, cell_x: int, cell_y: int) -> Tuple[float, float]:
        """Convert map coordinates to world coordinates."""
        x = (cell_x - self.config.origin_x) * self.config.resolution
        y = (cell_y - self.config.origin_y) * self.config.resolution
        return x, y

    def is_valid_cell(self, cell_x: int, cell_y: int) -> bool:
        """Check if cell coordinates are within map bounds."""
        return (0 <= cell_x < self.config.width and
                0 <= cell_y < self.config.height)

    def update_from_depth(
        self,
        depth_map: np.ndarray,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
        max_range: float = 3.0,     # Maximum sensing range in meters
        fov: float = 60.0,          # Field of view in degrees
        height_filter: bool = True,  # Only consider floor-level obstacles
    ):
        """
        Update occupancy map from depth observation.

        Args:
            depth_map: Relative depth (0-1, where 1=near, 0=far)
            pose_x, pose_y: Drone position in meters
            pose_yaw: Drone heading in radians
            max_range: Maximum depth sensing range
            fov: Camera field of view
            height_filter: If True, focus on middle rows (floor level)
        """
        h, w = depth_map.shape

        # Get drone position in map coordinates
        drone_cx, drone_cy = self.world_to_map(pose_x, pose_y)

        # Record trajectory
        if self.is_valid_cell(drone_cx, drone_cy):
            self.trajectory.append((drone_cx, drone_cy))
            # Mark drone position as free
            self._update_cell(drone_cx, drone_cy, free=True)

        # Sample rays across the field of view
        fov_rad = np.radians(fov)
        n_rays = 60  # Number of rays to cast

        # Focus on middle rows for floor-level obstacles
        if height_filter:
            row_start = int(h * 0.4)
            row_end = int(h * 0.7)
        else:
            row_start = 0
            row_end = h

        for i in range(n_rays):
            # Angle for this ray (relative to forward)
            angle_offset = (i / (n_rays - 1) - 0.5) * fov_rad
            ray_angle = pose_yaw + angle_offset

            # Get column in depth image
            col = int((i / (n_rays - 1)) * (w - 1))

            # Get depth for this column (average over height range)
            col_depths = depth_map[row_start:row_end, col]

            # Use minimum depth (closest obstacle)
            if len(col_depths) > 0:
                # Depth is inverted: 1 = near, 0 = far
                # Robustify against single-pixel spikes/noise:
                # Use a high percentile (still "near"-biased) instead of max().
                rel_depth = float(np.percentile(col_depths, 90))
                rel_depth = float(np.clip(rel_depth, 0.0, 1.0))

                if rel_depth > 0.1:  # Ignore very far / unknown
                    # Convert relative depth to distance
                    # rel_depth=1 -> dist≈0, rel_depth=0 -> dist=max_range
                    distance = (1.0 - rel_depth) * max_range
                    # Clamp: avoid marking obstacles unrealistically close due to noise
                    distance = max(distance, 0.25)

                    # Calculate obstacle position
                    obs_x = pose_x + distance * np.cos(ray_angle)
                    obs_y = pose_y + distance * np.sin(ray_angle)

                    obs_cx, obs_cy = self.world_to_map(obs_x, obs_y)

                    # Ray trace: mark cells along ray as free
                    self._ray_trace(drone_cx, drone_cy, obs_cx, obs_cy,
                                   mark_end_occupied=(rel_depth > 0.55))

    def _ray_trace(
        self,
        x0: int, y0: int,
        x1: int, y1: int,
        mark_end_occupied: bool = True
    ):
        """
        Trace a ray from (x0,y0) to (x1,y1).
        Mark cells along ray as free, and optionally end cell as occupied.

        Uses Bresenham's line algorithm.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        if dx > dy:
            err = dx / 2
            while x != x1:
                if self.is_valid_cell(x, y):
                    self._update_cell(x, y, free=True)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2
            while y != y1:
                if self.is_valid_cell(x, y):
                    self._update_cell(x, y, free=True)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        # Mark end cell
        if self.is_valid_cell(x1, y1):
            if mark_end_occupied:
                self._update_cell(x1, y1, free=False)
            else:
                self._update_cell(x1, y1, free=True)

    def _update_cell(self, x: int, y: int, free: bool, source: str = 'depth'):
        """
        Update a cell's occupancy.

        Args:
            x, y: Cell coordinates
            free: True = mark as free, False = mark as obstacle
            source: 'depth' for depth-sensed, 'collision' for collision-confirmed
        """
        if not self.is_valid_cell(x, y):
            return

        # Convert to int to avoid uint8 overflow before min/max
        current = int(self.grid[y, x])

        # CRITICAL: Don't overwrite collision-locked obstacles
        # (high visit count + low value = locked obstacle from collision detection)
        if current < 10 and self.visit_count[y, x] >= 5:
            # This cell was collision-confirmed - only collision can update it
            if source != 'collision':
                return

        if free:
            # Increase free confidence
            # But don't clear depth-sensed obstacles too aggressively
            if current < self.config.depth_obstacle and source == 'depth':
                # Don't let depth free-carving clear other depth obstacles
                return
            new_val = min(255, current + self.config.free_increment)
            self.visit_count[y, x] += 1
        else:
            # Mark as obstacle - different handling for depth vs collision
            if source == 'collision':
                # Collision-confirmed: strong, goes to 0
                new_val = max(0, current - self.config.occupied_increment)
                self.visit_count[y, x] += 3  # Higher confidence
            else:
                # Depth-sensed: weaker, floor at depth_obstacle level
                new_val = max(self.config.depth_obstacle, current - self.config.depth_occupied_increment)
                self.visit_count[y, x] += 1

        self.grid[y, x] = np.uint8(new_val)

    def mark_obstacle_ahead(
        self,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
        distance: float = 0.3,  # Distance ahead to mark as obstacle (meters)
    ):
        """
        Mark an obstacle directly ahead when a movement is blocked.

        This is called when the simulator blocks a forward movement,
        indicating there's a wall/obstacle that depth sensing missed.

        COLLISION-CONFIRMED obstacles get 3x3 dilation and are locked.

        Args:
            pose_x, pose_y: Current drone position in meters
            pose_yaw: Current heading in radians
            distance: Distance ahead to mark as obstacle
        """
        # Mark obstacles at 2 distances ahead (0.25m, 0.35m)
        for dist in [0.25, 0.35]:
            obs_x = pose_x + dist * np.cos(pose_yaw)
            obs_y = pose_y + dist * np.sin(pose_yaw)

            obs_cx, obs_cy = self.world_to_map(obs_x, obs_y)

            # Mark 3x3 area for collision-confirmed obstacles
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    cell_x = obs_cx + dx
                    cell_y = obs_cy + dy
                    if self.is_valid_cell(cell_x, cell_y):
                        # Use collision source - force to 0 and lock
                        self.grid[cell_y, cell_x] = 0
                        self.visit_count[cell_y, cell_x] += 10  # High confidence = locked

        print(f"  [MAP] Marked collision obstacle at ({pose_x:.2f},{pose_y:.2f}) yaw={np.degrees(pose_yaw):.0f}deg")

    def mark_place(self, place_id: int, x: float, y: float, label: str = None):
        """
        Mark a discovered place on the map.

        Args:
            place_id: Unique ID for this place (CSCG node)
            x, y: World coordinates in meters
            label: Optional semantic label (e.g., room type)
        """
        cx, cy = self.world_to_map(x, y)
        if self.is_valid_cell(cx, cy):
            self.places[place_id] = (cx, cy)
            if label:
                self.place_labels[place_id] = label

    def render(self, size: int = 300) -> np.ndarray:
        """
        Render the occupancy map as a color image.

        Returns:
            BGR image for display
        """
        # Create color image
        # Unknown = gray, Free = white, Occupied = black
        img = cv2.cvtColor(self.grid, cv2.COLOR_GRAY2BGR)

        # Color unknown slightly blue
        unknown_mask = self.grid == self.config.unknown
        img[unknown_mask] = [140, 130, 120]  # Grayish-blue

        # Draw trajectory in green
        if len(self.trajectory) > 1:
            pts = np.array(self.trajectory, dtype=np.int32)
            cv2.polylines(img, [pts], False, (0, 200, 0), 1)

        # Draw places as colored circles with semantic labels
        # Color by room type for semantic consistency
        room_colors = {
            "living_room": (255, 150, 100),   # Light blue
            "kitchen": (100, 200, 255),       # Orange
            "bedroom": (255, 100, 255),       # Magenta
            "bathroom": (255, 255, 100),      # Cyan
            "hallway": (150, 150, 150),       # Gray
            "unknown": (100, 100, 100),       # Dark gray
        }
        default_colors = [
            (255, 100, 100),  # Light blue
            (100, 255, 100),  # Light green
            (100, 100, 255),  # Light red
            (255, 255, 100),  # Cyan
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Yellow
        ]

        for place_id, (cx, cy) in self.places.items():
            # Get label and color
            label = self.place_labels.get(place_id, "")
            label_lower = label.lower().replace(" ", "_") if label else ""

            if label_lower in room_colors:
                color = room_colors[label_lower]
            else:
                color = default_colors[place_id % len(default_colors)]

            # Draw place marker
            cv2.circle(img, (cx, cy), 5, color, -1)
            cv2.circle(img, (cx, cy), 5, (255, 255, 255), 1)  # White border

            # Draw label (abbreviated room name + ID)
            if label:
                # Abbreviate room names
                abbrev = {
                    "living_room": "LR", "Living Room": "LR",
                    "kitchen": "K", "Kitchen": "K",
                    "bedroom": "BR", "Bedroom": "BR",
                    "bathroom": "BA", "Bathroom": "BA",
                    "hallway": "H", "Hallway": "H",
                }
                short_label = abbrev.get(label, label[:2].upper())
                display_text = f"{short_label}{place_id}"
            else:
                display_text = str(place_id)

            cv2.putText(img, display_text, (cx + 6, cy - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Draw current position (last trajectory point) as red dot
        if self.trajectory:
            cx, cy = self.trajectory[-1]
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        # Resize if needed
        if size != self.config.width:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)

        # Add border and label
        cv2.rectangle(img, (0, 0), (size-1, size-1), (100, 100, 100), 1)
        cv2.putText(img, "Occupancy Map", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Add scale bar
        scale_pixels = int(1.0 / self.config.resolution * size / self.config.width)
        cv2.line(img, (size - scale_pixels - 10, size - 10),
                (size - 10, size - 10), (255, 255, 255), 2)
        cv2.putText(img, "1m", (size - scale_pixels - 5, size - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Add legend for room types (compact, at bottom-left)
        if self.place_labels:
            legend_y = size - 60
            legend_items = [
                ("LR", room_colors["living_room"]),
                ("K", room_colors["kitchen"]),
                ("BR", room_colors["bedroom"]),
                ("BA", room_colors["bathroom"]),
                ("H", room_colors["hallway"]),
            ]
            cv2.putText(img, "Rooms:", (5, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            x_offset = 5
            for abbrev, color in legend_items:
                cv2.circle(img, (x_offset + 5, legend_y + 12), 3, color, -1)
                cv2.putText(img, abbrev, (x_offset + 10, legend_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
                x_offset += 25

        return img

    def get_stats(self) -> Dict:
        """Get map statistics."""
        total_cells = self.config.width * self.config.height
        unknown_cells = np.sum(self.grid == self.config.unknown)
        free_cells = np.sum(self.grid > self.config.unknown + 20)
        occupied_cells = np.sum(self.grid < self.config.unknown - 20)

        return {
            'total_cells': total_cells,
            'unknown_pct': 100 * unknown_cells / total_cells,
            'free_pct': 100 * free_cells / total_cells,
            'occupied_pct': 100 * occupied_cells / total_cells,
            'explored_pct': 100 * (free_cells + occupied_cells) / total_cells,
            'places_mapped': len(self.places),
            'trajectory_length': len(self.trajectory),
        }

    def reset(self):
        """Reset map to unknown."""
        self.grid.fill(self.config.unknown)
        self.visit_count.fill(0)
        self.trajectory.clear()
        self.places.clear()
        self.place_labels.clear()


class SpatialMapper:
    """
    High-level spatial mapping interface.

    Combines odometry and occupancy grid to build a spatial map
    of the environment as the drone explores.
    """

    def __init__(
        self,
        map_size: int = 200,
        resolution: float = 0.1,  # 10cm per cell
        use_visual_odometry: bool = False,  # Use action-based by default
    ):
        """
        Initialize spatial mapper.

        Args:
            map_size: Map size in cells
            resolution: Meters per cell
            use_visual_odometry: Use VO instead of action-based odometry
        """
        from .visual_odometry import ActionBasedOdometry, MonocularVO

        # Occupancy map
        self.map = OccupancyMap(MapConfig(
            width=map_size,
            height=map_size,
            resolution=resolution,
            origin_x=map_size // 2,
            origin_y=map_size // 2,
        ))

        # Odometry
        self.use_vo = use_visual_odometry
        if use_visual_odometry:
            self.odometry = MonocularVO()
        else:
            self.odometry = ActionBasedOdometry(
                move_distance=resolution * 3,  # ~3 cells per move
                turn_angle=15.0,
            )

        self.frame_count = 0

    def update(
        self,
        frame: np.ndarray,
        action: int,
        moved: bool,
        depth_map: np.ndarray = None,
        place_id: int = None,
        place_label: str = None,
    ):
        """
        Update spatial map with new observation.

        Args:
            frame: Current camera frame (BGR)
            action: Action taken (0-4)
            moved: Whether movement succeeded
            depth_map: Depth estimation (optional)
            place_id: Current place ID for marking (CSCG node)
            place_label: Semantic label for the place (e.g., room type)
        """
        self.frame_count += 1

        # Update odometry
        if self.use_vo:
            pose = self.odometry.update(frame)
            # Also update from action as fallback
            self.odometry.update_from_action(action, moved)
        else:
            pose = self.odometry.update(action, moved)

        # Update occupancy map from depth
        if depth_map is not None:
            self.map.update_from_depth(
                depth_map,
                pose.x, pose.y, pose.yaw,
                max_range=3.0,
            )

        # Mark current place with semantic label
        if place_id is not None:
            self.map.mark_place(place_id, pose.x, pose.y, label=place_label)

    def render(self, size: int = 200) -> np.ndarray:
        """Render the spatial map."""
        return self.map.render(size)

    def get_pose(self) -> Tuple[float, float, float]:
        """Get current pose (x, y, yaw)."""
        pose = self.odometry.pose
        return pose.x, pose.y, pose.yaw

    def get_stats(self) -> Dict:
        """Get mapping statistics."""
        stats = self.map.get_stats()
        pose = self.odometry.pose
        stats['pose_x'] = pose.x
        stats['pose_y'] = pose.y
        stats['pose_yaw_deg'] = np.degrees(pose.yaw)
        return stats

    def reset(self):
        """Reset mapper."""
        self.map.reset()
        self.odometry.reset()
        self.frame_count = 0
