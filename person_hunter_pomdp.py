"""
POMDP-based person hunting for Tello drone.

This is the main integration script that combines:
- World Model POMDP (localization)
- Human Search POMDP (person tracking)
- Interaction Mode POMDP (action selection)
- Safety Module (battery, collision overrides)

The drone learns a topological map of its environment while searching
for and interacting with people.

Usage:
    python person_hunter_pomdp.py              # Normal flight mode
    python person_hunter_pomdp.py --ground-test  # Ground test (no flight)
"""

import os
import time
import argparse
import cv2
import numpy as np
from typing import Optional, Dict, Tuple
from djitellopy import Tello
from ultralytics import YOLO

# POMDP modules
from pomdp import (
    WorldModel,
    HumanSearchPOMDP,
    InteractionModePOMDP,
    ExplorationModePOMDP,
    encode_yolo_detections,
    observation_to_text,
    action_to_rc_control,
    exploration_action_to_rc_control,
    load_latest_map,
    ACTION_LAND,
    ACTION_BACKOFF,
    ACTIONS,
)
from pomdp.config import MAX_FB, MAX_YAW, SEARCH_YAW

# CSCG + VBGS modules
from mapping.cscg import CSCGWorldModel
from mapping.vbgs_place import PlaceManager

# Safety module
from safety import SafetyMonitor, SafetyState, SafetyOverride

# Utilities
from utils import FrameGrabber, clamp
from utils.spatial_map import SpatialMap, combine_with_map

# Suppress djitellopy logging
import logging
logging.getLogger("djitellopy").setLevel(logging.WARNING)


# =============================================================================
# Configuration
# =============================================================================

STREAM_URL = "udp://0.0.0.0:11111"
YOLO_IMG_SIZE = 320
CONF_MIN = 0.45
UP_AFTER_TAKEOFF_CM = 40

# Visualization colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (100, 100, 100)


# =============================================================================
# POMDP Controller
# =============================================================================

class POMDPController:
    """
    Integrates all POMDP layers with mode switching between exploration and hunting.

    Modes:
        - exploration: Prioritizes building topological map via scanning + frontier exploration
        - hunting: Searches for and interacts with people

    Data flow:
        YOLO frame -> ObservationToken -> WorldModel -> [Exploration | HumanSearch+Interaction] -> RC Control
    """

    def __init__(
        self,
        load_existing_map: bool = True,
        start_in_exploration: bool = True,
        use_cscg: bool = False
    ):
        """
        Initialize all POMDPs.

        Args:
            load_existing_map: If True, try to load most recent saved map
            start_in_exploration: If True, start in exploration mode
            use_cscg: If True, use CSCG backend instead of TopologicalMap
        """
        self.use_cscg = use_cscg

        # Initialize world model based on backend choice
        self.world_model = None
        self.cscg_model = None

        if use_cscg:
            # Use CSCG backend
            print("Initializing CSCG (Clone-Structured Cognitive Graph) backend...")
            self.cscg_model = CSCGWorldModel(
                n_clones_per_token=10,  # Clones per observation token
                n_tokens=64,            # Max discrete observation tokens
                embedding_dim=512,      # CLIP embedding dimension
            )
            # Create a minimal WorldModel for compatibility
            self.world_model = WorldModel()
            # Spatial map for visualization
            self.spatial_map = SpatialMap(width=400, height=400)
            print("CSCG backend initialized")
        else:
            # Use standard TopologicalMap backend
            self.spatial_map = None
            if load_existing_map:
                try:
                    self.world_model = WorldModel.load()
                    if self.world_model is not None:
                        print(f"Loaded existing map with {self.world_model.n_locations} locations")
                except Exception as e:
                    print(f"Could not load existing map: {e}")

            if self.world_model is None:
                self.world_model = WorldModel()
                print("Starting with fresh world model")
            else:
                # Print info about loaded locations
                print("Loaded locations:")
                for loc_id in range(self.world_model.n_locations):
                    info = self.world_model.get_location_info(loc_id)
                    objs = ", ".join([o['name'] for o in info.get('top_objects', [])[:3]]) or "no objects"
                    print(f"  Loc {loc_id}: [{objs}] visits={info.get('visit_count', 0)}")

        # Initialize other POMDPs
        n_locs = max(1, self.world_model.n_locations)
        self.human_search = HumanSearchPOMDP(n_locations=n_locs)
        self.interaction = InteractionModePOMDP()
        self.exploration = ExplorationModePOMDP()

        # Mode management
        self.mode: str = "exploration" if start_in_exploration else "hunting"
        self._mode_locked: bool = False

        # Track last action for world model transition learning
        self._last_action_idx: int = 0

    def update(
        self,
        yolo_boxes,
        model_names: Dict[int, str],
        frame_width: int,
        frame_height: int,
        safety_override: Optional[int] = None,
        frame: np.ndarray = None,
    ) -> Dict:
        """
        Run full POMDP update cycle based on current mode.

        Args:
            yolo_boxes: YOLO detection boxes
            model_names: YOLO class name mapping
            frame_width: Frame width for normalization
            frame_height: Frame height for normalization
            safety_override: Optional safety override action index
            frame: Optional BGR image frame for CLIP embedding extraction

        Returns:
            Dict with all results and diagnostics
        """
        # 1. Encode YOLO detections to fixed observation token
        obs = encode_yolo_detections(
            yolo_boxes, model_names,
            frame_width, frame_height,
            conf_threshold=CONF_MIN
        )

        # 2. World Model: localize and update location belief
        if self.use_cscg and self.cscg_model is not None:
            # Use CSCG backend for localization
            loc_result = self.cscg_model.localize(
                frame=frame,
                action_taken=self._last_action_idx,
                observation_token=obs,
            )
            # Update spatial map for visualization
            if self.spatial_map is not None:
                self.spatial_map.update(loc_result.token, self._last_action_idx)
            # Also update the basic world model for compatibility
            _ = self.world_model.localize(obs, action_taken=self._last_action_idx, frame=frame)
        else:
            # Use standard TopologicalMap backend
            loc_result = self.world_model.localize(obs, action_taken=self._last_action_idx, frame=frame)

        # 3. Handle location expansion in HumanSearch
        if loc_result.new_location_discovered:
            n_locs = self.cscg_model.n_clone_states if self.use_cscg else self.world_model.n_locations
            self.human_search.expand_to_locations(n_locs)

        # 4. Mode-specific update
        if self.mode == "exploration":
            return self._exploration_update(obs, loc_result, safety_override)
        else:
            return self._hunting_update(obs, loc_result, safety_override)

    def _exploration_update(
        self,
        obs,
        loc_result,
        safety_override: Optional[int] = None
    ) -> Dict:
        """Run exploration mode update."""
        # Update exploration POMDP
        explore_result = self.exploration.update(obs, self.world_model, loc_result)

        # Check for automatic transition to hunting
        if not self._mode_locked and explore_result.should_transition_to_hunt:
            print(f"Transitioning to HUNTING mode: {explore_result.transition_reason}")
            self.mode = "hunting"
            self.interaction.reset_to_searching()

        # Get RC control
        if safety_override is not None:
            # For safety, use backoff action
            if safety_override == ACTION_BACKOFF:
                lr, fb, ud, yaw = (0, -MAX_FB, 5, 0)
            else:
                lr, fb, ud, yaw = (0, 0, 0, 0)
        else:
            lr, fb, ud, yaw = exploration_action_to_rc_control(explore_result.selected_action)

        # Track action for world model transition learning
        self._last_action_idx = explore_result.selected_action

        return {
            'observation': obs,
            'localization': loc_result,
            'exploration': explore_result,
            'mode': 'exploration',
            'rc_control': (lr, fb, ud, yaw),
            'action_name': explore_result.selected_action_name,
        }

    def _hunting_update(
        self,
        obs,
        loc_result,
        safety_override: Optional[int] = None
    ) -> Dict:
        """Run hunting mode update (original behavior)."""
        # Human Search: update belief over human locations
        human_result = self.human_search.update(obs, drone_location=loc_result.location_id)

        # Interaction Mode: select action (with optional safety override)
        if safety_override is not None:
            interaction_result = self.interaction.update_with_action_override(
                obs, override_action=safety_override
            )
        else:
            interaction_result = self.interaction.update(obs)

        # Track action for next iteration's transition
        self._last_action_idx = interaction_result.selected_action_idx

        # Convert action to RC control
        lr, fb, ud, yaw = action_to_rc_control(
            interaction_result.selected_action_idx,
            person_cx=obs.person_cx,
            person_area=obs.person_area,
            max_fb=MAX_FB,
            max_yaw=MAX_YAW,
            search_yaw=SEARCH_YAW,
        )

        return {
            'observation': obs,
            'localization': loc_result,
            'human_search': human_result,
            'interaction': interaction_result,
            'mode': 'hunting',
            'rc_control': (lr, fb, ud, yaw),
            'action_name': interaction_result.selected_action,
        }

    def set_mode(self, mode: str, lock: bool = False) -> None:
        """
        Manually set operating mode.

        Args:
            mode: 'exploration' or 'hunting'
            lock: If True, prevent automatic mode transitions
        """
        if mode in ("exploration", "hunting"):
            self.mode = mode
            self._mode_locked = lock
            print(f"Mode set to: {mode} (locked={lock})")

            if mode == "exploration":
                self.exploration.reset()
            else:
                self.interaction.reset_to_searching()

    def save_map(self, name: str = None) -> str:
        """Save the learned world model map."""
        return self.world_model.save(name=name)

    def render_spatial_map(self) -> Optional[np.ndarray]:
        """Render the spatial map (CSCG only)."""
        if self.spatial_map is not None:
            return self.spatial_map.render()
        return None

    def get_diagnostics(self) -> Dict:
        """Get diagnostic info from all POMDPs."""
        human_stats = self.human_search.get_statistics()
        interaction_stats = self.interaction.get_statistics()

        diag = {
            'mode': self.mode,
            'world_model': {
                'n_locations': self.world_model.n_locations,
                'current_location': self.world_model.current_location_id,
                'confidence': self.world_model.confidence,
                'belief_entropy': self.world_model.get_belief_entropy(),
            },
            'human_search': {
                'most_likely_location': human_stats['max_belief_location'],
                'last_sighting_location': human_stats['last_sighting_location'],
                'belief_entropy': human_stats['belief_entropy'],
            },
            'interaction': {
                'engagement_state': interaction_stats['most_likely_state'],
                'last_action': interaction_stats['last_action'],
                'confidence': interaction_stats['confidence'],
            },
        }

        if self.mode == "exploration":
            diag['exploration'] = self.exploration.get_statistics()

        # Add CSCG-specific diagnostics
        if self.use_cscg and self.cscg_model is not None:
            diag['cscg'] = {
                'n_tokens': self.cscg_model.tokenizer.n_tokens if hasattr(self.cscg_model, 'tokenizer') else 0,
                'n_clone_states': self.cscg_model.n_clone_states,
                'current_clone': self.cscg_model.current_clone_state,
                'current_token': self.cscg_model.current_token,
                'n_places': self.cscg_model.place_manager.n_places if hasattr(self.cscg_model, 'place_manager') else 0,
            }

        return diag


# =============================================================================
# Visualization
# =============================================================================

def draw_pomdp_overlay(
    frame: np.ndarray,
    result: Dict,
    safety: SafetyState
) -> np.ndarray:
    """
    Draw POMDP belief and status overlay on frame.

    Args:
        frame: BGR frame to draw on
        result: POMDP update result dict
        safety: Current safety state

    Returns:
        Frame with overlay
    """
    h, w = frame.shape[:2]
    mode = result.get('mode', 'hunting')

    # Mode indicator (top right)
    mode_color = COLOR_YELLOW if mode == "exploration" else COLOR_GREEN
    cv2.putText(frame, f"MODE: {mode.upper()}", (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    # Battery indicator (top right, below mode)
    bat_color = COLOR_GREEN if safety.battery_level > 20 else COLOR_YELLOW if safety.battery_level > 10 else COLOR_RED
    cv2.putText(frame, f"BAT: {safety.battery_level}%", (w - 150, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bat_color, 2)

    # Location info (top left)
    loc = result['localization']

    # Check if this is CSCG result (has token attribute)
    if hasattr(loc, 'token'):
        # CSCG mode: show token and clone state
        cv2.putText(frame, f"TKN: {loc.token} CLONE: {loc.clone_state} ({loc.confidence:.0%})", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    else:
        # Standard mode: show location
        cv2.putText(frame, f"LOC: {loc.location_id} ({loc.confidence:.0%}) sim={loc.similarity:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

    # New location indicator
    if loc.new_location_discovered:
        indicator_text = "NEW TOKEN!" if hasattr(loc, 'token') else "NEW LOCATION!"
        cv2.putText(frame, indicator_text, (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2)

    # Mode-specific overlay
    if mode == "exploration":
        _draw_exploration_overlay(frame, result, h, w)
    else:
        _draw_hunting_overlay(frame, result, h, w)

    # Safety warnings (center)
    if safety.contact_detected:
        cv2.putText(frame, "CONTACT!", (w // 2 - 60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_RED, 3)

    if safety.battery_warning and not safety.battery_critical:
        cv2.putText(frame, "LOW BATTERY", (w // 2 - 80, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)

    if safety.battery_critical:
        cv2.putText(frame, "BATTERY CRITICAL - LANDING", (w // 2 - 150, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)

    # RC control indicator (bottom right)
    rc = result['rc_control']
    cv2.putText(frame, f"RC: fb={rc[1]} yaw={rc[3]}", (w - 200, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

    return frame


def _draw_exploration_overlay(frame: np.ndarray, result: Dict, h: int, w: int) -> None:
    """Draw exploration mode specific overlay."""
    explore = result['exploration']

    # Exploration state
    state_color = {
        'scanning': COLOR_YELLOW,
        'approaching_frontier': COLOR_GREEN,
        'backtracking': COLOR_YELLOW,
        'transitioning': COLOR_WHITE,
    }.get(explore.exploration_state, COLOR_WHITE)
    cv2.putText(frame, f"STATE: {explore.exploration_state}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

    # Selected action
    cv2.putText(frame, f"ACTION: {explore.selected_action_name}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)

    # Locations discovered
    cv2.putText(frame, f"Locations: {explore.locations_discovered}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

    # VFE stats (bottom left)
    cv2.putText(frame, f"VFE: {explore.current_vfe:.2f} (mean: {explore.mean_vfe:.2f})", (20, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    cv2.putText(frame, f"VFE variance: {explore.vfe_variance:.3f}", (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

    # Transition readiness bar
    bar_width = min(300, w - 40)
    bar_height = 20
    bar_x = 20
    bar_y = h - 30

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

    # Fill based on VFE (lower VFE = more ready to transition)
    # Map VFE to progress (VFE 5.0 = 0%, VFE 0.5 = 100%)
    from pomdp.config import EXPLORATION_VFE_THRESHOLD
    progress = max(0.0, min(1.0, 1.0 - (explore.mean_vfe / 5.0)))
    fill_width = int(progress * bar_width)
    fill_color = COLOR_GREEN if explore.should_transition_to_hunt else COLOR_YELLOW
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), fill_color, -1)

    cv2.putText(frame, "Model fit:", (bar_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)


def _draw_hunting_overlay(frame: np.ndarray, result: Dict, h: int, w: int) -> None:
    """Draw hunting mode specific overlay."""
    interaction = result['interaction']
    human = result['human_search']

    # Engagement state
    state_color = {
        'searching': COLOR_YELLOW,
        'approaching': COLOR_GREEN,
        'interacting': COLOR_GREEN,
        'disengaging': COLOR_YELLOW,
    }.get(interaction.engagement_state, COLOR_WHITE)
    cv2.putText(frame, f"STATE: {interaction.engagement_state}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

    # Selected action
    cv2.putText(frame, f"ACTION: {interaction.selected_action}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)

    # Human search info (bottom left)
    if human.person_detected:
        cv2.putText(frame, "PERSON DETECTED", (20, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)

    cv2.putText(frame, f"Search target: Loc {human.search_target}", (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

    # Draw belief distribution as bar
    belief = np.array(human.belief)
    n_locs = len(belief) - 1  # Exclude "not visible"
    if n_locs > 0:
        bar_width = min(300, w - 40)
        bar_height = 20
        bar_x = 20
        bar_y = h - 30

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        # Draw belief segments
        x = bar_x
        for i in range(n_locs):
            seg_width = int(float(belief[i]) * bar_width)
            if seg_width > 1:
                color = COLOR_GREEN if i == human.most_likely_location else COLOR_GRAY
                cv2.rectangle(frame, (x, bar_y), (x + seg_width, bar_y + bar_height), color, -1)
            x += seg_width

        # Label
        cv2.putText(frame, "Human belief:", (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with full POMDP integration."""

    # Parse arguments
    parser = argparse.ArgumentParser(description="POMDP-based Tello person hunter")
    parser.add_argument(
        "--ground-test",
        action="store_true",
        help="Ground test mode: stream video and build map without flying"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the map on exit"
    )
    parser.add_argument(
        "--cscg",
        action="store_true",
        help="Use CSCG (Clone-Structured Cognitive Graph) backend for world model"
    )
    parser.add_argument(
        "--simulate",
        type=str,
        nargs='?',
        const='webcam',
        default=None,
        help="Simulate with webcam (default) or video file path. No drone needed."
    )
    args = parser.parse_args()

    # Check for YOLO weights
    weights_path = "./yolov8n.pt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing YOLO weights: {weights_path}")

    tello = Tello()
    airborne = False
    model = YOLO(weights_path)
    grabber = None
    cap = None

    try:
        # Connect to Tello
        print("Connecting to Tello...")
        tello.connect()
        print(f"Battery: {tello.get_battery()}%")

        # Initialize safety monitor
        safety = SafetyMonitor(tello)

        # Initialize POMDP controller (loads existing map if available)
        pomdp = POMDPController(load_existing_map=True, use_cscg=args.cscg)

        # Start video stream
        try:
            tello.streamoff()
        except Exception:
            pass
        time.sleep(0.5)
        tello.streamon()
        time.sleep(2.0)

        # Open video capture
        cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video stream")

        # Wait for first frame
        print("Waiting for video stream...")
        t0 = time.time()
        while time.time() - t0 < 10.0:
            ret, frame = cap.read()
            if ret and frame is not None:
                print("Got first frame!")
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Failed to grab first frame within 10s")

        # Start threaded frame grabber
        grabber = FrameGrabber(cap)

        # Create display window
        cv2.namedWindow("POMDP Hunter", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("POMDP Hunter", 960, 720)

        # Warm up YOLO
        print("Warming up YOLO...")
        for _ in range(10):
            frame = grabber.get_frame()
            if frame is not None:
                model(frame, imgsz=YOLO_IMG_SIZE, verbose=False)

        # Pre-load CLIP if using CSCG (to avoid blocking during flight)
        if args.cscg and pomdp.cscg_model is not None:
            print("Pre-loading CLIP model...")
            frame = grabber.get_frame()
            if frame is not None:
                # This triggers lazy loading of the image encoder
                _ = pomdp.cscg_model._get_image_encoder()
                # Do a warmup encode
                pomdp.cscg_model._get_image_encoder().encode(frame)
            print("CLIP ready.")

        # === GROUND TEST MODE ===
        if args.ground_test:
            print("\n=== GROUND TEST MODE ===")
            print("Carry the drone around to test map learning.")
            print("Controls: E = Exploration, H = Hunting, S = Save map, Q = Quit")
            print("No flight commands will be sent.\n")

            frame_count = 0
            while True:
                frame = grabber.get_frame()
                if frame is None:
                    continue

                frame_count += 1
                h, w = frame.shape[:2]

                # Get safety state (for battery display)
                safety_state, _ = safety.update()

                # YOLO detection
                res = model(frame, imgsz=YOLO_IMG_SIZE, verbose=False)[0]

                # POMDP update (no safety override in ground test)
                result = pomdp.update(
                    yolo_boxes=res.boxes,
                    model_names=model.names,
                    frame_width=w,
                    frame_height=h,
                    safety_override=None,
                    frame=frame,  # Pass frame for CLIP embedding
                )

                # Draw overlay
                overlay_frame = draw_pomdp_overlay(frame.copy(), result, safety_state)

                # Add ground test indicator
                cv2.putText(overlay_frame, "GROUND TEST", (w // 2 - 80, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Combine with spatial map if using CSCG
                spatial_map_img = pomdp.render_spatial_map()
                if spatial_map_img is not None:
                    display_frame = combine_with_map(overlay_frame, spatial_map_img)
                else:
                    display_frame = overlay_frame

                cv2.imshow("POMDP Hunter", display_frame)

                # Log every 30 frames
                if frame_count % 30 == 0:
                    diag = pomdp.get_diagnostics()
                    loc_info = pomdp.world_model.get_location_info()
                    top_objs = ", ".join([o['name'] for o in loc_info.get('top_objects', [])[:3]])

                    # CSCG-specific logging
                    if 'cscg' in diag:
                        cscg = diag['cscg']
                        if result['mode'] == 'exploration':
                            explore_stats = diag.get('exploration', {})
                            print(f"[{frame_count}] Mode=EXPLORE CSCG, "
                                  f"Token={cscg['current_token']}, "
                                  f"Clone={cscg['current_clone']}/{cscg['n_clone_states']}, "
                                  f"Tokens={cscg['n_tokens']}, "
                                  f"Places={cscg['n_places']}, "
                                  f"VFE={explore_stats.get('mean_vfe', 0):.2f}")
                        else:
                            print(f"[{frame_count}] Mode=HUNT CSCG, "
                                  f"Token={cscg['current_token']}, "
                                  f"Clone={cscg['current_clone']}, "
                                  f"State={diag['interaction']['engagement_state']}")
                    else:
                        # Standard TopologicalMap logging
                        if result['mode'] == 'exploration':
                            explore_stats = diag.get('exploration', {})
                            print(f"[{frame_count}] Mode=EXPLORE, "
                                  f"Loc={diag['world_model']['current_location']} [{top_objs}], "
                                  f"Locations={diag['world_model']['n_locations']}, "
                                  f"State={explore_stats.get('state', 'unknown')}, "
                                  f"VFE={explore_stats.get('mean_vfe', 0):.2f}")
                        else:
                            print(f"[{frame_count}] Mode=HUNT, "
                                  f"Loc={diag['world_model']['current_location']} [{top_objs}], "
                                  f"State={diag['interaction']['engagement_state']}")

                # Keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    print("Quit requested.")
                    break
                elif key in (ord('e'), ord('E')):
                    pomdp.set_mode("exploration")
                    print("Switched to EXPLORATION mode")
                elif key in (ord('h'), ord('H')):
                    pomdp.set_mode("hunting")
                    print("Switched to HUNTING mode")
                elif key in (ord('s'), ord('S')):
                    try:
                        map_path = pomdp.save_map()
                        print(f"Map saved to: {map_path}")
                    except Exception as e:
                        print(f"Failed to save map: {e}")

            # Save map on exit (unless --no-save)
            if not args.no_save:
                print("Saving map...")
                try:
                    map_path = pomdp.save_map()
                    print(f"Map saved to: {map_path}")
                except Exception as e:
                    print(f"Failed to save map: {e}")

            return  # Exit without flight cleanup

        # === NORMAL FLIGHT MODE ===
        print("Ready! Press T to takeoff, Q to quit.")
        print("During flight: E = Exploration mode, H = Hunting mode")

        # Pre-flight loop - wait for takeoff command
        while True:
            frame = grabber.get_frame()
            if frame is not None:
                # Show preview with instructions
                cv2.putText(frame, "Press T to TAKEOFF", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2)
                cv2.putText(frame, "Press Q to QUIT", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_YELLOW, 2)

                # Show battery status
                state, _ = safety.update()
                cv2.putText(frame, f"Battery: {state.battery_level}%", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

                cv2.imshow("POMDP Hunter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('t'), ord('T')):
                break
            if key in (ord('q'), ord('Q')):
                print("Quitting without takeoff.")
                return

        # Takeoff
        print("TAKEOFF...")
        tello.takeoff()
        time.sleep(1.0)
        airborne = True

        # Rise to operating altitude
        try:
            tello.move_up(UP_AFTER_TAKEOFF_CM)
        except Exception as e:
            print(f"Move up failed: {e}")
        time.sleep(1.0)

        # Flush old frames
        for _ in range(30):
            grabber.get_frame()
            time.sleep(0.03)

        # === MAIN CONTROL LOOP ===
        print("Running POMDP control loop...")
        frame_count = 0
        last_fb = 0  # Track for contact detection

        while True:
            frame = grabber.get_frame()
            if frame is None:
                continue

            frame_count += 1
            h, w = frame.shape[:2]

            # 1. SAFETY CHECK FIRST (outside POMDP)
            safety_state, safety_override = safety.update(commanded_fb=last_fb)

            # Handle critical safety conditions
            if safety_state.emergency:
                print(f"EMERGENCY: {safety_state.emergency_reason}")
                break

            # Map safety override to POMDP action index
            override_action_idx = None
            if safety_override == SafetyOverride.BACKOFF:
                override_action_idx = ACTION_BACKOFF
            elif safety_override in (SafetyOverride.LAND, SafetyOverride.EMERGENCY_LAND):
                override_action_idx = ACTION_LAND

            # 2. YOLO DETECTION
            res = model(frame, imgsz=YOLO_IMG_SIZE, verbose=False)[0]

            # 3. POMDP UPDATE CYCLE
            result = pomdp.update(
                yolo_boxes=res.boxes,
                model_names=model.names,
                frame_width=w,
                frame_height=h,
                safety_override=override_action_idx,
                frame=frame,  # Pass frame for CLIP embedding
            )

            # 4. EXECUTE ACTION
            lr, fb, ud, yaw = result['rc_control']
            tello.send_rc_control(lr, fb, ud, yaw)
            last_fb = fb  # Track for next iteration's contact detection

            # 5. VISUALIZATION
            frame = draw_pomdp_overlay(frame, result, safety_state)

            # Draw YOLO detections
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names.get(cls_id, "")

                if label == "person":
                    color = COLOR_GREEN
                else:
                    color = COLOR_WHITE

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.0%}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Combine with spatial map if using CSCG
            spatial_map_img = pomdp.render_spatial_map()
            if spatial_map_img is not None:
                display_frame = combine_with_map(frame, spatial_map_img)
            else:
                display_frame = frame

            cv2.imshow("POMDP Hunter", display_frame)

            # 6. LOGGING (periodic)
            if frame_count % 60 == 0:
                diag = pomdp.get_diagnostics()
                if result['mode'] == 'exploration':
                    explore_stats = diag.get('exploration', {})
                    print(f"[{frame_count}] Mode=EXPLORE, "
                          f"Loc={diag['world_model']['current_location']}, "
                          f"State={explore_stats.get('state', 'unknown')}, "
                          f"VFE={explore_stats.get('mean_vfe', 0):.2f}, "
                          f"Battery={safety_state.battery_level}%")
                else:
                    print(f"[{frame_count}] Mode=HUNT, "
                          f"Loc={diag['world_model']['current_location']}, "
                          f"State={diag['interaction']['engagement_state']}, "
                          f"Action={result['action_name']}, "
                          f"Battery={safety_state.battery_level}%")

            # 7. CHECK KEYBOARD INPUT
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("Quit requested.")
                break
            elif key in (ord('e'), ord('E')):
                pomdp.set_mode("exploration")
            elif key in (ord('h'), ord('H')):
                pomdp.set_mode("hunting")

            # Check for landing action selected by POMDP (only in hunting mode)
            if result['mode'] == 'hunting':
                if result['interaction'].selected_action_idx == ACTION_LAND:
                    print("POMDP selected LAND action.")
                    break

        # Landing
        print("Landing...")
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.3)
        tello.land()
        airborne = False

        # Save learned map
        print("Saving map...")
        try:
            map_path = pomdp.save_map()
            print(f"Map saved to: {map_path}")
        except Exception as e:
            print(f"Failed to save map: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("Cleaning up...")

        try:
            if grabber:
                grabber.stop()
        except Exception:
            pass

        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.streamoff()
        except Exception:
            pass

        try:
            if airborne:
                print("Emergency landing...")
                tello.land()
            tello.end()
        except Exception:
            pass

        try:
            if cap:
                cap.release()
        except Exception:
            pass

        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
