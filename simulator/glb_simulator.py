"""
GLB-based 3D simulator for testing place recognition.

Loads a GLB house model and renders first-person views.
Uses pyrender for high-quality offscreen rendering.
"""

import cv2
import numpy as np
import math
import os
from pathlib import Path
from typing import Tuple, Optional


class GLBSimulator:
    """
    First-person 3D simulator using a GLB house model.

    Uses pyrender for rendering real 3D models with textures.
    """

    def __init__(
        self,
        glb_path: str = None,
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
    ):
        """
        Initialize simulator with GLB model.

        Args:
            glb_path: Path to GLB file (defaults to simple_house.glb)
            width: Render width
            height: Render height
            fov: Field of view in degrees
        """
        self.width = width
        self.height = height
        self.fov = fov

        # Find model path
        if glb_path is None:
            glb_path = Path(__file__).parent / "home_-_p3.glb"
        self.glb_path = Path(glb_path)

        # Camera state - start in middle of house, looking along X
        # Model bounds: ~930 x 616 x 1490 units
        # Center roughly at (10, 290, -600) based on bounds
        self.x = 0.0      # X position
        self.y = 200.0    # Y position (up)
        self.z = -400.0   # Z position (forward)
        self.yaw = 0.0    # Horizontal rotation (radians)
        self.pitch = 0.0  # Vertical rotation (radians)

        # Movement settings
        self.move_speed = 30.0  # Units per step (model is ~1000 units)
        self.turn_speed = math.radians(15)

        # Room definitions - will be set after loading model
        self.rooms = {}

        # Collision avoidance state
        self._prev_frame: np.ndarray = None
        self._collision_ema: float = 0.0
        self._consecutive_blocked: int = 0

        # Load model (also sets up room definitions based on model bounds)
        self._load_model()

    def _load_model(self):
        """Load GLB model and setup pyrender scene."""
        import trimesh
        import pyrender

        print(f"Loading model from {self.glb_path}...")

        # Load GLB
        self.trimesh_scene = trimesh.load(str(self.glb_path))

        # Get bounds for positioning
        bounds = self.trimesh_scene.bounds
        self.model_center = (bounds[0] + bounds[1]) / 2
        self.model_extents = self.trimesh_scene.extents
        print(f"Model center: {self.model_center}")
        print(f"Model extents: {self.model_extents}")

        # Create pyrender scene - filter out non-mesh geometry (Path3D, PointCloud, etc.)
        self.scene = pyrender.Scene()

        # Add meshes from trimesh scene with their transforms
        mesh_count = 0
        skip_count = 0

        if hasattr(self.trimesh_scene, 'graph') and hasattr(self.trimesh_scene, 'geometry'):
            # Scene with graph - iterate through nodes to get transforms
            for node_name in self.trimesh_scene.graph.nodes_geometry:
                try:
                    # Get the transform and geometry name for this node
                    transform, geom_name = self.trimesh_scene.graph[node_name]
                    geom = self.trimesh_scene.geometry.get(geom_name)

                    if geom is None:
                        continue

                    # Only process actual meshes
                    if isinstance(geom, trimesh.Trimesh):
                        mesh = pyrender.Mesh.from_trimesh(geom)
                        self.scene.add(mesh, pose=transform)
                        mesh_count += 1
                    else:
                        skip_count += 1
                except Exception as e:
                    skip_count += 1

        elif hasattr(self.trimesh_scene, 'geometry'):
            # Fallback: add meshes without transforms
            for name, geom in self.trimesh_scene.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    try:
                        mesh = pyrender.Mesh.from_trimesh(geom)
                        self.scene.add(mesh)
                        mesh_count += 1
                    except Exception as e:
                        skip_count += 1
                else:
                    skip_count += 1

        elif isinstance(self.trimesh_scene, trimesh.Trimesh):
            # Single mesh
            mesh = pyrender.Mesh.from_trimesh(self.trimesh_scene)
            self.scene.add(mesh)
            mesh_count = 1

        print(f"Loaded {mesh_count} meshes (skipped {skip_count} non-mesh objects)")

        # Add ambient light
        self.scene.ambient_light = np.array([0.4, 0.4, 0.4])

        # Add directional light for shadows
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, :3] = self._rotation_matrix(math.radians(-45), math.radians(30), 0)
        self.scene.add(light, pose=light_pose)

        # Add point light at camera position
        self.camera_light = pyrender.PointLight(color=[1.0, 0.95, 0.9], intensity=1000.0)
        self.camera_light_node = self.scene.add(self.camera_light)

        # Create camera
        self.camera = pyrender.PerspectiveCamera(
            yfov=math.radians(self.fov),
            aspectRatio=self.width / self.height,
            znear=0.1,
            zfar=5000.0
        )
        self.camera_node = self.scene.add(self.camera)

        # Create offscreen renderer
        self.renderer = pyrender.OffscreenRenderer(self.width, self.height)

        # Position camera in a specific room at comfortable drone height
        # Y is vertical axis - start at ~50% of room height (eye level)
        bounds = self.trimesh_scene.bounds
        floor_y = bounds[0][1]  # Bottom of model (floor level)
        ceiling_y = bounds[1][1]  # Top of model
        room_height = ceiling_y - floor_y

        # Start at 50% of room height (comfortable flying height)
        drone_height = floor_y + room_height * 0.5

        # Set up rooms first so we can start in a specific room
        self._setup_rooms()

        # Start in the kitchen (center of kitchen bounds)
        if "Kitchen" in self.rooms:
            (x_min, x_max), (z_min, z_max) = self.rooms["Kitchen"]
            self.x = (x_min + x_max) / 2
            self.z = (z_min + z_max) / 2
        else:
            # Fallback to model center
            self.x = self.model_center[0]
            self.z = self.model_center[2]

        self.y = drone_height

        print(f"Starting position: ({self.x:.1f}, {self.y:.1f}, {self.z:.1f}) in {self._get_current_room()}")
        print(f"Floor Y: {floor_y:.1f}, Ceiling Y: {ceiling_y:.1f}, Drone height: {drone_height:.1f}")

        print("Model loaded successfully!")

    def _setup_rooms(self):
        """
        Set up room definitions based on model bounds.

        Divides the model into a grid of rooms for navigation tracking.
        """
        bounds = self.trimesh_scene.bounds
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]

        # Add margin to keep drone inside
        margin = 50
        x_min += margin
        x_max -= margin
        z_min += margin
        z_max -= margin

        # Store navigable bounds (including Y for floor/ceiling)
        # Keep drone between 30% and 70% of room height
        y_floor = y_min + (y_max - y_min) * 0.25
        y_ceiling = y_min + (y_max - y_min) * 0.70
        self._nav_bounds = (x_min, x_max, y_floor, y_ceiling, z_min, z_max)

        # Divide into 2x3 grid of rooms (typical house layout)
        x_mid = (x_min + x_max) / 2
        z_third = (z_max - z_min) / 3

        z_front = z_max
        z_mid1 = z_max - z_third
        z_mid2 = z_max - 2 * z_third
        z_back = z_min

        # Room definitions: (x_range, z_range)
        self.rooms = {
            "Living Room": ((x_min, x_mid), (z_mid1, z_front)),
            "Kitchen": ((x_mid, x_max), (z_mid1, z_front)),
            "Hallway": ((x_min, x_max), (z_mid2, z_mid1)),
            "Bedroom": ((x_min, x_mid), (z_back, z_mid2)),
            "Bathroom": ((x_mid, x_max), (z_back, z_mid2)),
        }

        print(f"Room layout configured:")
        for name, ((xlo, xhi), (zlo, zhi)) in self.rooms.items():
            print(f"  {name}: x=[{xlo:.0f}, {xhi:.0f}], z=[{zlo:.0f}, {zhi:.0f}]")
        print(f"Vertical bounds: Y=[{y_floor:.0f}, {y_ceiling:.0f}] (floor to ceiling)")

    def _check_ray_collision(self, action: int, min_distance: float = 50.0) -> bool:
        """
        Check for obstacles using ray casting (accurate collision detection).

        Casts a ray from the drone's position in the direction of movement
        to detect walls and obstacles. Much more accurate than depth buffer.

        Args:
            action: Movement action (1=forward, 2=back, 5=strafe left, 6=strafe right)
            min_distance: Minimum distance (in scene units) to consider blocked

        Returns:
            True if path is blocked, False if clear
        """
        import trimesh

        # Get movement direction based on action
        if action == 1:  # Forward
            dx = math.sin(self.yaw)
            dz = -math.cos(self.yaw)
        elif action == 2:  # Backward
            dx = -math.sin(self.yaw)
            dz = math.cos(self.yaw)
        elif action == 5:  # Strafe left
            dx = -math.cos(self.yaw)
            dz = -math.sin(self.yaw)
        elif action == 6:  # Strafe right
            dx = math.cos(self.yaw)
            dz = math.sin(self.yaw)
        else:
            return False  # Non-translation action

        # Ray origin (current position)
        origin = np.array([[self.x, self.y, self.z]])

        # Ray direction (normalized)
        direction = np.array([[dx, 0, dz]])

        # Cast ray against the scene
        # Get all meshes from trimesh scene for ray casting
        try:
            # Create a ray-mesh query
            if hasattr(self.trimesh_scene, 'ray'):
                # Scene has built-in ray casting
                locations, index_ray, index_tri = self.trimesh_scene.ray.intersects_location(
                    ray_origins=origin,
                    ray_directions=direction
                )
            else:
                # Fallback: iterate through meshes
                closest_hit = float('inf')
                for name, geom in self.trimesh_scene.geometry.items():
                    if isinstance(geom, trimesh.Trimesh):
                        try:
                            locations, index_ray, index_tri = geom.ray.intersects_location(
                                ray_origins=origin,
                                ray_directions=direction
                            )
                            if len(locations) > 0:
                                distances = np.linalg.norm(locations - origin, axis=1)
                                closest_hit = min(closest_hit, np.min(distances))
                        except:
                            pass

                if closest_hit < min_distance:
                    return True
                return False

            if len(locations) == 0:
                return False  # No hit, path is clear

            # Find closest intersection
            distances = np.linalg.norm(locations - origin, axis=1)
            closest_distance = np.min(distances)

            # Blocked if closest hit is within min_distance
            return closest_distance < min_distance

        except Exception as e:
            # Fallback to bounds check if ray casting fails
            return False

    def _check_depth_obstacle(self, action: int, min_distance: float = 80.0) -> bool:
        """
        Check for obstacles using the depth buffer (backup method).

        This method renders the current view and checks if there's an obstacle
        too close in the direction of movement. Works like a real depth sensor.

        Args:
            action: Movement action (1=forward, 2=back, 5=strafe left, 6=strafe right)
            min_distance: Minimum distance (in scene units) to consider blocked

        Returns:
            True if path is blocked, False if clear
        """
        # Use ray casting for accurate collision detection
        if self._check_ray_collision(action, min_distance):
            return True

        # Fallback: depth buffer check for forward movement only
        if action != 1:
            return False

        # Render current view to get depth buffer
        self._update_camera()
        color, depth = self.renderer.render(self.scene)

        h, w = depth.shape
        center_y = h // 2
        center_x = w // 2

        # Check center region for forward movement
        y1, y2 = max(0, center_y - 40), min(h, center_y + 40)
        x1, x2 = max(0, center_x - 80), min(w, center_x + 80)

        sample_region = depth[y1:y2, x1:x2]

        if sample_region.size == 0:
            return False

        # Find closest object (minimum non-zero depth)
        valid_mask = (sample_region > 0) & (sample_region < 10000)
        valid_depths = sample_region[valid_mask]

        if len(valid_depths) == 0:
            return False

        closest_distance = np.min(valid_depths)

        return closest_distance < min_distance

    def get_depth_buffer(self) -> np.ndarray:
        """
        Get the current depth buffer from the camera view.

        Returns:
            Depth image where values represent distance from camera.
        """
        self._update_camera()
        color, depth = self.renderer.render(self.scene)
        return depth

    def _rotation_matrix(self, yaw: float, pitch: float, roll: float = 0) -> np.ndarray:
        """Create rotation matrix from Euler angles."""
        # Yaw (Y-axis rotation)
        cy, sy = math.cos(yaw), math.sin(yaw)
        # Pitch (X-axis rotation)
        cp, sp = math.cos(pitch), math.sin(pitch)
        # Roll (Z-axis rotation)
        cr, sr = math.cos(roll), math.sin(roll)

        Ry = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, cp, -sp],
            [0, sp, cp]
        ])

        Rz = np.array([
            [cr, -sr, 0],
            [sr, cr, 0],
            [0, 0, 1]
        ])

        return Ry @ Rx @ Rz

    def _update_camera(self):
        """Update camera pose in scene."""
        import pyrender

        # Create camera transformation matrix
        pose = np.eye(4)

        # Rotation
        pose[:3, :3] = self._rotation_matrix(self.yaw, self.pitch)

        # Translation
        pose[0, 3] = self.x
        pose[1, 3] = self.y
        pose[2, 3] = self.z

        # Update camera node
        self.scene.set_pose(self.camera_node, pose)

        # Update camera light position
        self.scene.set_pose(self.camera_light_node, pose)

    def move(self, action: int, debug: bool = False) -> bool:
        """
        Move camera based on action.

        Args:
            action: 0=stay, 1=forward, 2=back, 3=turn left, 4=turn right,
                   5=strafe left, 6=strafe right, 7=look up, 8=look down
            debug: Print debug info

        Returns:
            True if movement succeeded, False if blocked by boundary
        """
        action_names = ['stay', 'forward', 'back', 'left', 'right',
                       'strafe_left', 'strafe_right', 'look_up', 'look_down']

        # Save old position for collision check
        old_x, old_z = self.x, self.z

        # Check depth-based obstacle detection BEFORE moving (like real drone)
        obstacle_ahead = False
        if action in (1, 2, 5, 6):  # All translation actions
            obstacle_ahead = self._check_depth_obstacle(action)

        moved = True

        if action == 0:  # Stay
            pass
        elif action == 1:  # Forward
            if obstacle_ahead:
                moved = False
                if debug:
                    print(f"  [GLB] BLOCKED by wall (depth sensor)!")
            else:
                self.x += self.move_speed * math.sin(self.yaw)
                self.z -= self.move_speed * math.cos(self.yaw)
        elif action == 2:  # Back
            if obstacle_ahead:
                moved = False
                if debug:
                    print(f"  [GLB] BLOCKED by wall (depth sensor)!")
            else:
                self.x -= self.move_speed * math.sin(self.yaw)
                self.z += self.move_speed * math.cos(self.yaw)
        elif action == 3:  # Turn left
            self.yaw += self.turn_speed
        elif action == 4:  # Turn right
            self.yaw -= self.turn_speed
        elif action == 5:  # Strafe left
            if obstacle_ahead:
                moved = False
                if debug:
                    print(f"  [GLB] BLOCKED by wall on left (depth sensor)!")
            else:
                self.x -= self.move_speed * math.cos(self.yaw)
                self.z -= self.move_speed * math.sin(self.yaw)
        elif action == 6:  # Strafe right
            if obstacle_ahead:
                moved = False
                if debug:
                    print(f"  [GLB] BLOCKED by wall on right (depth sensor)!")
            else:
                self.x += self.move_speed * math.cos(self.yaw)
                self.z += self.move_speed * math.sin(self.yaw)
        elif action == 7:  # Look up
            self.pitch = min(self.pitch + self.turn_speed, math.radians(60))
        elif action == 8:  # Look down
            self.pitch = max(self.pitch - self.turn_speed, math.radians(-60))

        # Normalize yaw
        self.yaw = self.yaw % (2 * math.pi)

        # Check bounds - keep drone inside the model
        if action in (1, 2, 5, 6) and moved:  # Movement actions
            if not self._is_inside_bounds(self.x, self.z):
                # Revert movement
                self.x, self.z = old_x, old_z
                moved = False
                if debug:
                    print(f"  [GLB] BLOCKED at boundary!")

        if debug and moved:
            room = self._get_current_room()
            print(f"  [GLB] Moved {action_names[action]} to ({self.x:.0f},{self.y:.0f},{self.z:.0f}) yaw={math.degrees(self.yaw):.0f}° room={room}")

        return moved

    def _is_inside_bounds(self, x: float, z: float, y: float = None) -> bool:
        """
        Check if position is inside the navigable house area.

        Uses dynamically computed bounds from model geometry.
        """
        if y is None:
            y = self.y

        if hasattr(self, '_nav_bounds'):
            x_min, x_max, y_floor, y_ceiling, z_min, z_max = self._nav_bounds
        else:
            # Fallback to model bounds
            bounds = self.trimesh_scene.bounds
            x_min, x_max = bounds[0][0], bounds[1][0]
            y_floor, y_ceiling = bounds[0][1], bounds[1][1]
            z_min, z_max = bounds[0][2], bounds[1][2]

        return (x_min <= x <= x_max and
                y_floor <= y <= y_ceiling and
                z_min <= z <= z_max)

    def render(self) -> np.ndarray:
        """Render current view."""
        self._update_camera()

        # Render scene
        color, depth = self.renderer.render(self.scene)

        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # Draw room name
        room_name = self._get_current_room()
        cv2.putText(frame, room_name, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw minimap
        self._draw_minimap(frame)

        return frame

    def _get_current_room(self) -> str:
        """Determine current room based on position."""
        for room_name, ((x_min, x_max), (z_min, z_max)) in self.rooms.items():
            if x_min <= self.x <= x_max and z_min <= self.z <= z_max:
                return room_name
        return "Unknown"

    def _draw_minimap(self, frame: np.ndarray, size: int = 100):
        """Draw a small top-down minimap."""
        margin = 10
        map_x = self.width - size - margin
        map_y = margin

        # Background
        cv2.rectangle(frame, (map_x, map_y), (map_x + size, map_y + size),
                     (40, 40, 40), -1)
        cv2.rectangle(frame, (map_x, map_y), (map_x + size, map_y + size),
                     (100, 100, 100), 1)

        # Scale based on model extents
        scale = size / max(self.model_extents[0], self.model_extents[2]) * 0.8

        # Draw camera position
        center_x = map_x + size // 2
        center_y = map_y + size // 2

        cam_map_x = int(center_x + (self.x - self.model_center[0]) * scale)
        cam_map_y = int(center_y - (self.z - self.model_center[2]) * scale)

        cv2.circle(frame, (cam_map_x, cam_map_y), 3, (0, 255, 255), -1)

        # Draw direction
        dir_len = 10
        dir_x = int(cam_map_x + dir_len * math.sin(self.yaw))
        dir_y = int(cam_map_y - dir_len * math.cos(self.yaw))
        cv2.line(frame, (cam_map_x, cam_map_y), (dir_x, dir_y), (0, 255, 255), 2)

        # Draw position text
        pos_text = f"({self.x:.0f}, {self.z:.0f})"
        cv2.putText(frame, pos_text, (map_x, map_y + size + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def get_position(self) -> Tuple[float, float, float]:
        """Get current position and yaw angle."""
        return self.x, self.z, self.yaw

    def close(self):
        """Clean up renderer."""
        if hasattr(self, 'renderer'):
            self.renderer.delete()


def run_simulator_standalone():
    """Run the GLB simulator standalone for testing."""
    sim = GLBSimulator()

    print("\nGLB 3D Simulator")
    print("Controls:")
    print("  WASD / Arrow keys: Move")
    print("  Q/E: Strafe left/right")
    print("  R/F: Look up/down")
    print("  ESC: Quit")

    cv2.namedWindow("GLB Simulator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GLB Simulator", 960, 720)

    try:
        while True:
            frame = sim.render()
            cv2.imshow("GLB Simulator", frame)

            key = cv2.waitKey(30) & 0xFF

            if key in (27,):  # ESC
                break
            elif key in (ord('w'), ord('W'), 82):  # W or Up
                sim.move(1, debug=True)
            elif key in (ord('s'), ord('S'), 84):  # S or Down
                sim.move(2, debug=True)
            elif key in (ord('a'), ord('A'), 81):  # A or Left
                sim.move(3, debug=True)
            elif key in (ord('d'), ord('D'), 83):  # D or Right
                sim.move(4, debug=True)
            elif key in (ord('q'), ord('Q')):  # Strafe left
                sim.move(5, debug=True)
            elif key in (ord('e'), ord('E')):  # Strafe right
                sim.move(6, debug=True)
            elif key in (ord('r'), ord('R')):  # Look up
                sim.move(7, debug=True)
            elif key in (ord('f'), ord('F')):  # Look down
                sim.move(8, debug=True)

    finally:
        cv2.destroyAllWindows()
        sim.close()


if __name__ == "__main__":
    run_simulator_standalone()
