"""
Simple 3D room simulator for testing CSCG mapping.

Renders a first-person view of a simple house layout.
No external 3D libraries - just OpenCV and numpy.

Enhanced with furniture, decorations, and distinct textures for CLIP.
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Wall:
    """A wall segment defined by two endpoints."""
    x1: float
    y1: float
    x2: float
    y2: float
    color: Tuple[int, int, int]  # BGR
    texture_id: int = 0


@dataclass
class Furniture:
    """A piece of furniture (rendered as a box)."""
    x: float
    y: float
    width: float  # X dimension
    depth: float  # Y dimension
    height: float  # Z dimension (for rendering)
    color: Tuple[int, int, int]  # BGR
    name: str = ""


@dataclass
class Room:
    """A room with walls and a name."""
    name: str
    walls: List[Wall]
    floor_color: Tuple[int, int, int]
    center_x: float
    center_y: float
    furniture: List[Furniture] = field(default_factory=list)
    ceiling_color: Tuple[int, int, int] = (60, 50, 40)


class Simple3DSimulator:
    """
    Simple first-person 3D renderer for room navigation.

    Uses raycasting for wall rendering (like Wolfenstein 3D).
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
    ):
        """
        Initialize simulator.

        Args:
            width: Render width
            height: Render height
            fov: Field of view in degrees
        """
        self.width = width
        self.height = height
        self.fov = math.radians(fov)

        # Camera state
        self.x = 5.0  # Start in living room
        self.y = 5.0
        self.angle = 0.0  # Facing right (east)

        # Movement settings
        self.move_speed = 0.3
        self.turn_speed = math.radians(15)

        # Build the house
        self.walls: List[Wall] = []
        self.rooms: List[Room] = []
        self._build_house()

        # Textures (simple patterns)
        self._textures = self._create_textures()

    def _build_house(self):
        """Build a simple house layout with furniture."""

        # Colors for different rooms (BGR) - MORE DISTINCT
        LIVING_ROOM = (60, 100, 180)    # Warm orange-beige
        KITCHEN = (80, 180, 80)          # Bright green
        BEDROOM = (200, 120, 80)         # Cool blue
        BATHROOM = (200, 200, 80)        # Bright cyan/teal
        HALLWAY = (100, 100, 100)        # Neutral gray

        # Furniture colors
        SOFA_RED = (60, 60, 180)         # Red sofa
        TABLE_BROWN = (50, 80, 120)      # Brown table
        TV_BLACK = (30, 30, 30)          # Black TV
        COUNTER_WHITE = (200, 200, 200)  # White counter
        FRIDGE_SILVER = (180, 180, 190)  # Silver fridge
        BED_BLUE = (180, 120, 60)        # Blue bed
        DRESSER_WOOD = (60, 100, 140)    # Wood dresser
        TOILET_WHITE = (240, 240, 240)   # White toilet
        TUB_WHITE = (220, 230, 240)      # Bathtub

        # House layout (10x12 units):
        #
        #  +--------+--------+
        #  | BEDROOM| BATH   |
        #  |   (3)  |  (4)   |
        #  +----+---+---+----+
        #  |    HALLWAY (5)  |
        #  +----+-------+----+
        #  | LIVING  | KITCHEN|
        #  | ROOM(1) |   (2)  |
        #  +---------+--------+

        # Outer walls
        outer = [
            Wall(0, 0, 12, 0, HALLWAY, 0),     # Bottom
            Wall(12, 0, 12, 10, HALLWAY, 0),   # Right
            Wall(12, 10, 0, 10, HALLWAY, 0),   # Top
            Wall(0, 10, 0, 0, HALLWAY, 0),     # Left
        ]

        # Living room (bottom-left) with furniture
        living_room_walls = [
            Wall(0, 0, 7, 0, LIVING_ROOM, 1),
            Wall(7, 0, 7, 4, LIVING_ROOM, 1),
            Wall(7, 4, 0, 4, LIVING_ROOM, 1),
            Wall(0, 4, 0, 0, LIVING_ROOM, 1),
        ]
        living_furniture = [
            Furniture(1.5, 0.8, 2.5, 0.8, 0.4, SOFA_RED, "sofa"),      # Sofa against wall
            Furniture(1.5, 2.0, 1.0, 0.6, 0.3, TABLE_BROWN, "coffee_table"),  # Coffee table
            Furniture(5.5, 0.5, 0.3, 0.5, 0.8, TV_BLACK, "tv"),        # TV stand
        ]
        self.rooms.append(Room("Living Room", living_room_walls, (50, 70, 100), 3.5, 2,
                               living_furniture, (70, 60, 50)))

        # Kitchen (bottom-right) with appliances
        kitchen_walls = [
            Wall(7, 0, 12, 0, KITCHEN, 2),
            Wall(12, 0, 12, 4, KITCHEN, 2),
            Wall(12, 4, 7, 4, KITCHEN, 2),
            Wall(7, 4, 7, 0, KITCHEN, 2),
        ]
        kitchen_furniture = [
            Furniture(11.0, 1.0, 0.8, 2.5, 0.9, COUNTER_WHITE, "counter"),  # Counter
            Furniture(11.2, 0.5, 0.6, 0.6, 1.2, FRIDGE_SILVER, "fridge"),   # Fridge
            Furniture(8.0, 3.2, 1.5, 0.6, 0.9, COUNTER_WHITE, "island"),    # Kitchen island
        ]
        self.rooms.append(Room("Kitchen", kitchen_walls, (70, 100, 70), 9.5, 2,
                               kitchen_furniture, (60, 60, 50)))

        # Hallway (middle) - sparse
        hallway_walls = [
            Wall(0, 4, 12, 4, HALLWAY, 0),
            Wall(12, 4, 12, 6, HALLWAY, 0),
            Wall(12, 6, 0, 6, HALLWAY, 0),
            Wall(0, 6, 0, 4, HALLWAY, 0),
        ]
        hallway_furniture = [
            Furniture(1.0, 5.0, 0.4, 0.8, 0.5, TABLE_BROWN, "console"),  # Small console table
        ]
        self.rooms.append(Room("Hallway", hallway_walls, (80, 80, 80), 6, 5,
                               hallway_furniture, (50, 50, 50)))

        # Bedroom (top-left) with bed
        bedroom_walls = [
            Wall(0, 6, 6, 6, BEDROOM, 3),
            Wall(6, 6, 6, 10, BEDROOM, 3),
            Wall(6, 10, 0, 10, BEDROOM, 3),
            Wall(0, 10, 0, 6, BEDROOM, 3),
        ]
        bedroom_furniture = [
            Furniture(1.5, 8.5, 2.0, 1.5, 0.5, BED_BLUE, "bed"),          # Bed
            Furniture(4.5, 9.0, 0.8, 0.5, 0.7, DRESSER_WOOD, "dresser"),  # Dresser
            Furniture(0.5, 6.8, 0.4, 0.4, 0.5, TABLE_BROWN, "nightstand"), # Nightstand
        ]
        self.rooms.append(Room("Bedroom", bedroom_walls, (100, 90, 80), 3, 8,
                               bedroom_furniture, (80, 70, 60)))

        # Bathroom (top-right) with fixtures
        bathroom_walls = [
            Wall(6, 6, 12, 6, BATHROOM, 4),
            Wall(12, 6, 12, 10, BATHROOM, 4),
            Wall(12, 10, 6, 10, BATHROOM, 4),
            Wall(6, 10, 6, 6, BATHROOM, 4),
        ]
        bathroom_furniture = [
            Furniture(11.0, 7.0, 0.5, 0.5, 0.5, TOILET_WHITE, "toilet"),   # Toilet
            Furniture(7.5, 9.0, 2.0, 0.8, 0.5, TUB_WHITE, "bathtub"),      # Bathtub
            Furniture(10.5, 9.0, 0.6, 0.4, 0.8, COUNTER_WHITE, "sink"),    # Sink
        ]
        self.rooms.append(Room("Bathroom", bathroom_walls, (120, 120, 100), 9, 8,
                               bathroom_furniture, (100, 100, 90)))

        # Collect all walls (skip doorways)
        # Living room to hallway door (at x=3, y=4)
        # Kitchen to hallway door (at x=9, y=4)
        # Hallway to bedroom door (at x=3, y=6)
        # Hallway to bathroom door (at x=9, y=6)

        # Add walls with gaps for doors
        # Living room
        self.walls.append(Wall(0, 4, 2.5, 4, LIVING_ROOM, 1))
        self.walls.append(Wall(3.5, 4, 7, 4, LIVING_ROOM, 1))
        self.walls.append(Wall(7, 0, 7, 4, KITCHEN, 2))

        # Kitchen
        self.walls.append(Wall(7, 4, 8.5, 4, KITCHEN, 2))
        self.walls.append(Wall(9.5, 4, 12, 4, KITCHEN, 2))

        # Hallway to upper rooms
        self.walls.append(Wall(0, 6, 2.5, 6, BEDROOM, 3))
        self.walls.append(Wall(3.5, 6, 6, 6, BEDROOM, 3))
        self.walls.append(Wall(6, 6, 8.5, 6, BATHROOM, 4))
        self.walls.append(Wall(9.5, 6, 12, 6, BATHROOM, 4))

        # Room divider
        self.walls.append(Wall(6, 6, 6, 10, HALLWAY, 0))

        # Outer walls
        self.walls.extend(outer)

        # Set starting position (living room)
        self.x = 3.5
        self.y = 2.0
        self.angle = math.pi / 2  # Facing north

    def _create_textures(self) -> List[np.ndarray]:
        """Create distinct texture patterns for CLIP differentiation."""
        textures = []

        # Texture 0: Plain gray with subtle vertical gradient (hallway)
        tex = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            shade = 90 + (i * 40 // 64)
            tex[i, :] = [shade, shade, shade]
        textures.append(tex)

        # Texture 1: Bold vertical stripes (living room) - orange/brown
        tex = np.zeros((64, 64, 3), dtype=np.uint8)
        for j in range(64):
            if (j // 12) % 2 == 0:
                tex[:, j] = [40, 80, 180]   # Orange stripe
            else:
                tex[:, j] = [60, 100, 140]  # Brown stripe
        # Add a horizontal band for visual interest
        tex[28:36, :] = [80, 120, 200]
        textures.append(tex)

        # Texture 2: Checkerboard tiles (kitchen) - green/white
        tex = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                if ((i // 16) + (j // 16)) % 2 == 0:
                    tex[i, j] = [80, 200, 80]    # Bright green
                else:
                    tex[i, j] = [200, 220, 200]  # White
        textures.append(tex)

        # Texture 3: Horizontal wood grain (bedroom) - blue tones
        tex = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            if (i // 6) % 3 == 0:
                tex[i, :] = [220, 140, 80]   # Light blue
            elif (i // 6) % 3 == 1:
                tex[i, :] = [200, 120, 60]   # Medium blue
            else:
                tex[i, :] = [180, 100, 40]   # Dark blue
        textures.append(tex)

        # Texture 4: Small white tiles with grout (bathroom) - cyan/white
        tex = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                # Grout lines
                if i % 10 == 0 or j % 10 == 0:
                    tex[i, j] = [80, 80, 80]  # Dark grout
                else:
                    # Alternating tile colors
                    tile_i, tile_j = i // 10, j // 10
                    if (tile_i + tile_j) % 2 == 0:
                        tex[i, j] = [220, 220, 100]  # Cyan tile
                    else:
                        tex[i, j] = [240, 240, 240]  # White tile
        textures.append(tex)

        return textures

    def move(self, action: int, debug: bool = False) -> bool:
        """
        Move based on action.

        Args:
            action: 0=stay, 1=forward, 2=back, 3=turn left, 4=turn right
            debug: Print debug info

        Returns:
            True if movement succeeded
        """
        new_x, new_y = self.x, self.y
        action_names = ['stay', 'forward', 'back', 'left', 'right']
        moved = True

        if action == 1:  # Forward
            new_x = self.x + self.move_speed * math.cos(self.angle)
            new_y = self.y + self.move_speed * math.sin(self.angle)
        elif action == 2:  # Back
            new_x = self.x - self.move_speed * math.cos(self.angle)
            new_y = self.y - self.move_speed * math.sin(self.angle)
        elif action == 3:  # Turn left
            self.angle += self.turn_speed
        elif action == 4:  # Turn right
            self.angle -= self.turn_speed

        # Collision check
        if action in (1, 2):
            if self._collides(new_x, new_y):
                moved = False
                if debug:
                    print(f"  [SIM] BLOCKED! action={action_names[action]} pos=({self.x:.1f},{self.y:.1f}) angle={math.degrees(self.angle):.0f}deg")
            else:
                self.x, self.y = new_x, new_y
                if debug:
                    room = self._get_current_room()
                    print(f"  [SIM] Moved {action_names[action]} to ({self.x:.1f},{self.y:.1f}) room={room}")

        # Normalize angle
        self.angle = self.angle % (2 * math.pi)

        return moved

    def _collides(self, x: float, y: float, radius: float = 0.3) -> bool:
        """Check if position collides with any wall or furniture."""
        # Check walls
        for wall in self.walls:
            if self._point_near_segment(x, y, wall.x1, wall.y1, wall.x2, wall.y2, radius):
                return True

        # Check furniture
        for room in self.rooms:
            for furn in room.furniture:
                # Simple bounding box check with radius
                if (furn.x - furn.width/2 - radius <= x <= furn.x + furn.width/2 + radius and
                    furn.y - furn.depth/2 - radius <= y <= furn.y + furn.depth/2 + radius):
                    return True

        return False

    def _point_near_segment(
        self, px: float, py: float,
        x1: float, y1: float, x2: float, y2: float,
        threshold: float
    ) -> bool:
        """Check if point is within threshold of line segment."""
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy

        if length_sq == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2) < threshold

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        dist = math.sqrt((px - closest_x)**2 + (py - closest_y)**2)
        return dist < threshold

    def render(self) -> np.ndarray:
        """Render the current view with walls and furniture."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Get current room for colors
        current_room = self._get_current_room_obj()
        if current_room:
            ceiling_color = current_room.ceiling_color
            floor_color = current_room.floor_color
        else:
            ceiling_color = (60, 50, 40)
            floor_color = (50, 60, 70)

        # Draw ceiling and floor with room-specific colors
        frame[:self.height//2, :] = ceiling_color
        frame[self.height//2:, :] = floor_color

        # Depth buffer for furniture occlusion
        depth_buffer = np.full(self.width, float('inf'))

        # Raycasting for walls
        for col in range(self.width):
            ray_angle = self.angle + self.fov/2 - (col / self.width) * self.fov
            hit_dist, hit_wall, hit_u = self._cast_ray(ray_angle)

            if hit_dist > 0 and hit_wall is not None:
                hit_dist_corrected = hit_dist * math.cos(ray_angle - self.angle)
                depth_buffer[col] = hit_dist_corrected

                wall_height = min(self.height, int(self.height / (hit_dist_corrected + 0.1)))
                y_start = (self.height - wall_height) // 2
                y_end = y_start + wall_height

                tex = self._textures[hit_wall.texture_id]
                tex_x = int(hit_u * tex.shape[1]) % tex.shape[1]
                tex_col = tex[:, tex_x]

                if wall_height > 0:
                    tex_col_resized = cv2.resize(
                        tex_col.reshape(-1, 1, 3),
                        (1, wall_height),
                        interpolation=cv2.INTER_LINEAR
                    ).reshape(-1, 3)

                    shade = max(0.3, 1.0 - hit_dist_corrected / 10.0)
                    tex_col_resized = (tex_col_resized * shade).astype(np.uint8)

                    y_start = max(0, y_start)
                    y_end = min(self.height, y_end)
                    frame[y_start:y_end, col] = tex_col_resized[:y_end-y_start]

        # Render furniture as sprites/boxes
        self._render_furniture(frame, depth_buffer)

        # Draw current room name
        room_name = self._get_current_room()
        cv2.putText(frame, room_name, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw minimap
        self._draw_minimap(frame)

        return frame

    def _render_furniture(self, frame: np.ndarray, depth_buffer: np.ndarray):
        """Render furniture as 3D boxes."""
        # Collect all furniture with distances
        furniture_list = []
        for room in self.rooms:
            for furn in room.furniture:
                # Calculate distance to furniture center
                dx = furn.x - self.x
                dy = furn.y - self.y
                dist = math.sqrt(dx*dx + dy*dy)
                furniture_list.append((dist, furn))

        # Sort by distance (far to near for proper occlusion)
        furniture_list.sort(reverse=True, key=lambda x: x[0])

        for dist, furn in furniture_list:
            # Skip if too far
            if dist > 8.0:
                continue

            # Calculate angle to furniture
            dx = furn.x - self.x
            dy = furn.y - self.y
            angle_to_furn = math.atan2(dy, dx)

            # Calculate relative angle
            rel_angle = angle_to_furn - self.angle
            while rel_angle > math.pi:
                rel_angle -= 2 * math.pi
            while rel_angle < -math.pi:
                rel_angle += 2 * math.pi

            # Skip if not in field of view
            if abs(rel_angle) > self.fov / 2 + 0.3:
                continue

            # Calculate screen position
            screen_x = int(self.width / 2 - (rel_angle / self.fov) * self.width)

            # Calculate apparent size based on distance
            if dist < 0.3:
                dist = 0.3
            # Increased scaling for more visible furniture
            apparent_height = int(self.height * furn.height / dist * 1.2)
            apparent_width = int(self.width * max(furn.width, furn.depth) / dist * 0.4)

            # Clamp sizes
            apparent_height = min(apparent_height, self.height // 2)
            apparent_width = min(apparent_width, self.width // 3)

            if apparent_width < 2 or apparent_height < 2:
                continue

            # Calculate screen bounds
            x1 = screen_x - apparent_width // 2
            x2 = screen_x + apparent_width // 2
            y2 = self.height // 2 + apparent_height // 2  # Bottom
            y1 = y2 - apparent_height  # Top

            # Clamp to screen
            x1 = max(0, x1)
            x2 = min(self.width, x2)
            y1 = max(0, y1)
            y2 = min(self.height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Check depth buffer and draw
            for col in range(x1, x2):
                if depth_buffer[col] > dist:
                    # Draw this column of furniture
                    shade = max(0.4, 1.0 - dist / 6.0)
                    color = tuple(int(c * shade) for c in furn.color)
                    frame[y1:y2, col] = color

    def _get_current_room_obj(self) -> Optional[Room]:
        """Get the Room object the camera is in."""
        for room in self.rooms:
            walls = room.walls
            min_x = min(w.x1 for w in walls)
            max_x = max(w.x2 for w in walls)
            min_y = min(w.y1 for w in walls)
            max_y = max(w.y2 for w in walls)

            if min_x <= self.x <= max_x and min_y <= self.y <= max_y:
                return room
        return None

    def _cast_ray(self, angle: float) -> Tuple[float, Optional[Wall], float]:
        """
        Cast a ray and find the closest wall hit.

        Returns:
            (distance, wall, u) where u is the texture coordinate (0-1)
        """
        ray_dx = math.cos(angle)
        ray_dy = math.sin(angle)
        ray_length = 100.0  # Max ray distance

        closest_dist = float('inf')
        closest_wall = None
        closest_u = 0.0

        for wall in self.walls:
            # Line-line intersection
            x1, y1 = wall.x1, wall.y1
            x2, y2 = wall.x2, wall.y2

            x3, y3 = self.x, self.y
            x4, y4 = self.x + ray_dx * ray_length, self.y + ray_dy * ray_length

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                continue

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

            if 0 <= t <= 1 and u > 0:
                # Convert normalized parameter to actual distance
                dist = u * ray_length
                if dist < closest_dist:
                    closest_dist = dist
                    closest_wall = wall
                    closest_u = t

        if closest_dist == float('inf'):
            return -1, None, 0

        return closest_dist, closest_wall, closest_u

    def _get_current_room(self) -> str:
        """Get the name of the room the camera is in."""
        for room in self.rooms:
            # Simple check - is point in room's bounding area
            walls = room.walls
            min_x = min(w.x1 for w in walls)
            max_x = max(w.x2 for w in walls)
            min_y = min(w.y1 for w in walls)
            max_y = max(w.y2 for w in walls)

            if min_x <= self.x <= max_x and min_y <= self.y <= max_y:
                return room.name
        return "Unknown"

    def _draw_minimap(self, frame: np.ndarray, size: int = 100):
        """Draw a small minimap in the corner."""
        margin = 10
        map_x = self.width - size - margin
        map_y = margin

        # Background
        cv2.rectangle(frame, (map_x, map_y), (map_x + size, map_y + size),
                     (40, 40, 40), -1)
        cv2.rectangle(frame, (map_x, map_y), (map_x + size, map_y + size),
                     (100, 100, 100), 1)

        # Scale factor
        scale = size / 14.0  # House is 12x10, add margin
        offset_x = map_x + size // 2 - 6 * scale
        offset_y = map_y + size // 2 - 5 * scale

        # Draw walls
        for wall in self.walls:
            x1 = int(offset_x + wall.x1 * scale)
            y1 = int(offset_y + (10 - wall.y1) * scale)  # Flip Y
            x2 = int(offset_x + wall.x2 * scale)
            y2 = int(offset_y + (10 - wall.y2) * scale)
            cv2.line(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)

        # Draw camera position
        cam_x = int(offset_x + self.x * scale)
        cam_y = int(offset_y + (10 - self.y) * scale)
        cv2.circle(frame, (cam_x, cam_y), 3, (0, 255, 255), -1)

        # Draw direction
        dir_x = int(cam_x + 8 * math.cos(-self.angle + math.pi/2))
        dir_y = int(cam_y + 8 * math.sin(-self.angle + math.pi/2))
        cv2.line(frame, (cam_x, cam_y), (dir_x, dir_y), (0, 255, 255), 2)

    def get_position(self) -> Tuple[float, float, float]:
        """Get current position and angle."""
        return self.x, self.y, self.angle


def run_simulator_standalone():
    """Run the simulator standalone for testing."""
    sim = Simple3DSimulator()

    print("Simple 3D Simulator")
    print("Controls: WASD or Arrow keys to move, Q to quit")

    cv2.namedWindow("Simulator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Simulator", 960, 720)

    while True:
        frame = sim.render()
        cv2.imshow("Simulator", frame)

        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            break
        elif key in (ord('w'), ord('W'), 82):  # W or Up
            sim.move(1)
        elif key in (ord('s'), ord('S'), 84):  # S or Down
            sim.move(2)
        elif key in (ord('a'), ord('A'), 81):  # A or Left
            sim.move(3)
        elif key in (ord('d'), ord('D'), 83):  # D or Right
            sim.move(4)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_simulator_standalone()
