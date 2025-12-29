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
            glb_path = Path(__file__).parent / "simple_house.glb"
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

        # Room definitions based on model layout
        # These are approximations - adjust based on actual model
        self.rooms = {
            "Living Room": ((-200, 200), (-600, -200)),   # x range, z range
            "Kitchen": ((200, 400), (-600, -200)),
            "Bedroom": ((-200, 200), (-1200, -600)),
            "Bathroom": ((200, 400), (-1200, -600)),
            "Hallway": ((-50, 50), (-600, -200)),
        }

        # Load model
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

        # Create pyrender scene
        self.scene = pyrender.Scene.from_trimesh_scene(self.trimesh_scene)

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

        # Position camera at model center, above floor
        self.x = self.model_center[0]
        self.y = 200.0  # Eye height
        self.z = self.model_center[2]

        print("Model loaded successfully!")

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
            True if movement succeeded
        """
        action_names = ['stay', 'forward', 'back', 'left', 'right',
                       'strafe_left', 'strafe_right', 'look_up', 'look_down']

        if action == 0:  # Stay
            pass
        elif action == 1:  # Forward
            self.x += self.move_speed * math.sin(self.yaw)
            self.z -= self.move_speed * math.cos(self.yaw)
        elif action == 2:  # Back
            self.x -= self.move_speed * math.sin(self.yaw)
            self.z += self.move_speed * math.cos(self.yaw)
        elif action == 3:  # Turn left
            self.yaw += self.turn_speed
        elif action == 4:  # Turn right
            self.yaw -= self.turn_speed
        elif action == 5:  # Strafe left
            self.x -= self.move_speed * math.cos(self.yaw)
            self.z -= self.move_speed * math.sin(self.yaw)
        elif action == 6:  # Strafe right
            self.x += self.move_speed * math.cos(self.yaw)
            self.z += self.move_speed * math.sin(self.yaw)
        elif action == 7:  # Look up
            self.pitch = min(self.pitch + self.turn_speed, math.radians(60))
        elif action == 8:  # Look down
            self.pitch = max(self.pitch - self.turn_speed, math.radians(-60))

        # Normalize yaw
        self.yaw = self.yaw % (2 * math.pi)

        if debug:
            room = self._get_current_room()
            print(f"  [GLB] Moved {action_names[action]} to ({self.x:.0f},{self.y:.0f},{self.z:.0f}) yaw={math.degrees(self.yaw):.0f}° room={room}")

        return True

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
