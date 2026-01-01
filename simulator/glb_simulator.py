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

        # Movement settings - smaller steps = less likely to pass through walls
        self.move_speed = 15.0  # Units per step (reduced from 30 for safety)
        self.turn_speed = math.radians(10)  # Smaller turns for more precise navigation

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

        # Create a combined mesh for ray casting
        # This is needed because Scene objects don't have a direct ray tracer
        print("Setting up ray tracer for collision detection...")
        try:
            # Collect all meshes and combine them
            meshes = []
            if hasattr(self.trimesh_scene, 'geometry'):
                for name, geom in self.trimesh_scene.geometry.items():
                    if isinstance(geom, trimesh.Trimesh):
                        meshes.append(geom)

            if meshes:
                # Concatenate all meshes into one for ray casting
                self.collision_mesh = trimesh.util.concatenate(meshes)
                print(f"  Combined {len(meshes)} meshes for collision detection")
            else:
                self.collision_mesh = None
                print("  WARNING: No meshes found for collision detection!")
        except Exception as e:
            print(f"  WARNING: Failed to create collision mesh: {e}")
            self.collision_mesh = None

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

        # Start at 60% of room height (higher to avoid furniture)
        drone_height = floor_y + room_height * 0.6

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

        # Add margin to keep drone inside (reduced from 50 to allow more exploration)
        margin = 20
        x_min += margin
        x_max -= margin
        z_min += margin
        z_max -= margin

        # Store navigable bounds (including Y for floor/ceiling)
        # Keep drone between 40% and 75% of room height (higher to avoid furniture)
        y_floor = y_min + (y_max - y_min) * 0.40
        y_ceiling = y_min + (y_max - y_min) * 0.75
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

    def _check_ray_collision(self, action: int, min_distance: float = 80.0) -> bool:
        """
        Check for obstacles using MULTI-RAY fan (robust collision detection).

        Casts multiple rays in a wide fan to catch walls from any angle.
        Uses conservative distance threshold (2.5x move_speed).

        Args:
            action: Movement action (1=forward, 2=back, 5=strafe left, 6=strafe right)
            min_distance: Minimum distance (in scene units) to consider blocked
                         Default 80 = ~2.5x move_speed for safety margin

        Returns:
            True if path is blocked, False if clear
        """
        import trimesh

        # Get movement direction based on action
        if action == 1:  # Forward
            base_dx = math.sin(self.yaw)
            base_dz = -math.cos(self.yaw)
        elif action == 2:  # Backward - SHOULD NOT BE USED
            print(f"  [RAY] WARNING: Backward action detected at ({self.x:.0f}, {self.z:.0f})!")
            base_dx = -math.sin(self.yaw)
            base_dz = math.cos(self.yaw)
        elif action == 5:  # Strafe left
            base_dx = -math.cos(self.yaw)
            base_dz = -math.sin(self.yaw)
        elif action == 6:  # Strafe right
            base_dx = math.cos(self.yaw)
            base_dz = math.sin(self.yaw)
        else:
            return False  # Non-translation action

        try:
            # Use collision_mesh for ray casting (combined mesh with ray tracer)
            if self.collision_mesh is None:
                return False  # Can't check, allow movement

            # WIDE MULTI-RAY FAN: Cast 9 rays in a 60-degree fan
            fan_angles = [-0.52, -0.39, -0.26, -0.13, 0, 0.13, 0.26, 0.39, 0.52]
            origins = []
            directions = []

            for angle_offset in fan_angles:
                # Rotate direction by angle_offset in XZ plane
                cos_a = math.cos(angle_offset)
                sin_a = math.sin(angle_offset)
                dx = base_dx * cos_a - base_dz * sin_a
                dz = base_dx * sin_a + base_dz * cos_a

                origins.append([self.x, self.y, self.z])
                directions.append([dx, 0, dz])

            origins = np.array(origins)
            directions = np.array(directions)

            # Cast all rays at once using the collision mesh
            locations, index_ray, index_tri = self.collision_mesh.ray.intersects_location(
                ray_origins=origins,
                ray_directions=directions
            )

            if len(locations) == 0:
                # No hits - path is clear
                return False

            # Find closest hit across ALL rays
            closest_dist = float('inf')
            for i, loc in enumerate(locations):
                ray_idx = index_ray[i]
                dist = np.linalg.norm(loc - origins[ray_idx])
                if dist < closest_dist:
                    closest_dist = dist

            # Block if closest hit is too near
            if closest_dist < min_distance:
                return True

            return False

        except Exception as e:
            print(f"  [RAY] Error in collision check: {e}")
            return False  # If error, allow movement (was blocking everything)

    def _is_inside_geometry(self, min_clear_radius: float = 15.0) -> bool:
        """
        Check if the drone is inside geometry by casting rays in all directions.

        If ANY horizontal direction has a very close hit, we might be in a wall.
        Uses lower threshold (15 units = half move_speed) for safety.

        Args:
            min_clear_radius: Minimum distance that should be clear

        Returns:
            True if drone appears to be inside geometry
        """
        try:
            if self.collision_mesh is None:
                return False

            # Cast rays in 8 horizontal directions (more coverage)
            directions = []
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                directions.append([math.cos(rad), 0, math.sin(rad)])

            origin = np.array([self.x, self.y, self.z])
            origins = np.tile(origin, (len(directions), 1))
            directions = np.array(directions, dtype=np.float64)

            locations, index_ray, _ = self.collision_mesh.ray.intersects_location(
                ray_origins=origins,
                ray_directions=directions
            )

            if len(locations) == 0:
                # No hits at all - might be outside the model
                return False

            # Check if ANY direction has a very close hit
            for i, loc in enumerate(locations):
                ray_idx = index_ray[i]
                dist = np.linalg.norm(loc - origins[ray_idx])
                if dist < min_clear_radius:
                    return True

            return False

        except Exception as e:
            print(f"  [CONTAIN] Error: {e}")
            return False  # Allow movement if check fails

    def _check_depth_obstacle(self, action: int, min_distance: float = 18.0) -> bool:
        """
        Check for obstacles using DEPTH BUFFER (reliable, no rtree needed).

        Renders the current view and checks if there's an obstacle too close.
        Uses 10th percentile of depth to be robust against noise/outliers.

        Args:
            action: Movement action (1=forward, 5=strafe left, 6=strafe right)
            min_distance: Minimum distance (in scene units) to consider blocked
                         Default 25 = ~1.6x move_speed (15) for safety margin

        Returns:
            True if path is blocked, False if clear
        """
        # Only check for forward movement - strafe movements are protected by bounds check
        if action != 1:
            return False

        # Render current view to get depth buffer
        self._update_camera()
        try:
            color, depth = self.renderer.render(self.scene)
        except Exception as e:
            print(f"  [DEPTH] Render error: {e}")
            return False  # Allow movement on error

        h, w = depth.shape
        center_y = h // 2

        # Check WIDE horizontal strip (full width, but vertical center)
        # This catches walls at the edges of view, not just center
        y1, y2 = max(0, center_y - 30), min(h, center_y + 30)
        x1, x2 = 0, w  # Full width to catch walls at edges

        sample_region = depth[y1:y2, x1:x2]

        if sample_region.size == 0:
            return False

        # Find closest object using 5th percentile (catch walls at edges)
        valid_mask = (sample_region > 0) & (sample_region < 10000)
        valid_depths = sample_region[valid_mask]

        if len(valid_depths) < 50:  # Need some samples
            return False

        # Use 5th percentile - catches walls even at edges of view
        closest_distance = np.percentile(valid_depths, 5)

        # Block if too close
        if closest_distance < min_distance:
            return True

        return False

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

        # Rotation - flip yaw sign to match pyrender camera convention with motion
        # Motion uses sin(yaw)/cos(yaw) but pyrender camera forward is mirrored
        pose[:3, :3] = self._rotation_matrix(-self.yaw, self.pitch)

        # DEBUG: Compare camera forward vs motion forward
        # Motion forward: (sin(yaw), -cos(yaw)) in XZ plane
        # Camera forward from rotation matrix: -Z column (pose[:3, 2]) but negated
        motion_fwd = (math.sin(self.yaw), -math.cos(self.yaw))
        cam_fwd_xz = (-pose[0, 2], -pose[2, 2])  # Camera looks down -Z
        if hasattr(self, '_debug_frame_count'):
            self._debug_frame_count += 1
        else:
            self._debug_frame_count = 0
        if self._debug_frame_count % 100 == 0:
            print(f"  [CAM-DEBUG] yaw={math.degrees(self.yaw):.0f}deg "
                  f"motion_fwd=({motion_fwd[0]:.2f}, {motion_fwd[1]:.2f}) "
                  f"cam_fwd=({cam_fwd_xz[0]:.2f}, {cam_fwd_xz[1]:.2f})")

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
        action_names = ['stay', 'forward', 'back', 'turn_left', 'turn_right',
                       'strafe_left', 'strafe_right', 'look_up', 'look_down']

        # Save old position for collision check
        old_x, old_z = self.x, self.z
        old_room = self._get_current_room()

        # STRICT CHECK: Only allow movements where the camera is pointing
        # Block strafe (5, 6) - camera can't see sideways
        # Allow backward (2) for escape maneuvers when stuck
        if action in (5, 6):
            print(f"  [MOVE] BLOCKED: Strafe action rejected (camera can't see sideways)")
            return False

        # PRE-CHECK: Calculate new position and verify it's valid BEFORE moving
        if action == 1:  # Forward only (strafe and backward are blocked above)
            new_x = self.x + self.move_speed * math.sin(self.yaw)
            new_z = self.z - self.move_speed * math.cos(self.yaw)

            # Check if new position would be outside bounds
            if not self._is_inside_bounds(new_x, new_z):
                print(f"  [MOVE] PRE-BLOCKED: Would go outside bounds ({new_x:.0f}, {new_z:.0f})")
                return False

        # Check depth-based obstacle detection BEFORE moving (like real drone)
        obstacle_ahead = False
        if action == 1:  # Forward only
            obstacle_ahead = self._check_depth_obstacle(action)

        moved = True

        if action == 0:  # Stay
            pass
        elif action == 1:  # Forward
            if obstacle_ahead:
                moved = False
                print(f"  [MOVE] BLOCKED forward at ({self.x:.0f}, {self.z:.0f}) yaw={math.degrees(self.yaw):.0f}")
            else:
                self.x += self.move_speed * math.sin(self.yaw)
                self.z -= self.move_speed * math.cos(self.yaw)
        elif action == 2:  # Backward - for escape maneuvers
            # Check if backward is clear (simple bounds check, no ray check)
            old_x, old_z = self.x, self.z
            new_x = self.x - self.move_speed * math.sin(self.yaw)
            new_z = self.z + self.move_speed * math.cos(self.yaw)
            if self._is_inside_bounds(new_x, new_z):
                self.x = new_x
                self.z = new_z
                print(f"  [MOVE] Backward: ({old_x:.2f},{old_z:.2f}) -> ({self.x:.2f},{self.z:.2f}) yaw={math.degrees(self.yaw):.1f}")
            else:
                print(f"  [MOVE] BLOCKED backward at ({self.x:.2f},{self.z:.2f}) yaw={math.degrees(self.yaw):.1f}")
                moved = False
        elif action == 3:  # Turn left (CCW when viewed from above)
            self.yaw += self.turn_speed   # CCW increases yaw (standard math convention)
        elif action == 4:  # Turn right (CW when viewed from above)
            self.yaw -= self.turn_speed   # CW decreases yaw (standard math convention)
        # Strafe actions (5, 6) are blocked at the start of move() and will never reach here
        elif action == 7:  # Look up
            self.pitch = min(self.pitch + self.turn_speed, math.radians(60))
        elif action == 8:  # Look down
            self.pitch = max(self.pitch - self.turn_speed, math.radians(-60))

        # Normalize yaw
        self.yaw = self.yaw % (2 * math.pi)

        # Check bounds - keep drone inside the model
        if action == 1 and moved:  # Forward movement only
            new_room = self._get_current_room()

            # Check navigable bounds (should not happen due to pre-check, but double-check)
            if not self._is_inside_bounds(self.x, self.z):
                print(f"  [MOVE] REVERTED: Outside bounds ({self.x:.0f}, {self.z:.0f})")
                self.x, self.z = old_x, old_z
                moved = False
            # Log room transitions for debugging
            elif old_room != "Unknown" and new_room != old_room:
                print(f"  [MOVE] Room: {old_room} -> {new_room}")

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

    def get_state_snapshot(self) -> dict:
        """
        Get a snapshot of all mutable simulator state.
        Used for non-destructive escape scanning.
        """
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'yaw': self.yaw,
            'pitch': self.pitch,
            '_collision_ema': self._collision_ema,
            '_consecutive_blocked': self._consecutive_blocked,
        }

    def restore_state_snapshot(self, snapshot: dict):
        """
        Restore simulator state from a snapshot.
        Used for non-destructive escape scanning.
        """
        self.x = snapshot['x']
        self.y = snapshot['y']
        self.z = snapshot['z']
        self.yaw = snapshot['yaw']
        self.pitch = snapshot['pitch']
        self._collision_ema = snapshot['_collision_ema']
        self._consecutive_blocked = snapshot['_consecutive_blocked']

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
