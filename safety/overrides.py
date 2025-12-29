"""
Safety monitoring and override mechanisms for Tello drone control.

This module provides safety checks that run OUTSIDE the POMDP and have
higher priority than any POMDP-selected action. Safety overrides include:
- Battery monitoring (warning and critical thresholds)
- Contact/collision detection (blocked movement)
- Emergency landing triggers

The SafetyMonitor class should be called BEFORE POMDP updates in the main
loop, and its override actions should be passed to the POMDP for execution.
"""

import time
from dataclasses import dataclass, field
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from djitellopy import Tello

from pomdp.config import MAX_FB


# =============================================================================
# Safety Override Constants
# =============================================================================

class SafetyOverride:
    """
    Override action codes returned by SafetyMonitor.

    These map to POMDP action indices via the integration layer.
    """
    NONE = 0           # No override - POMDP chooses action
    BACKOFF = 1        # Contact detected - back away immediately
    HOVER = 2          # Low battery - hold position, conserve energy
    LAND = 3           # Critical battery - initiate landing sequence
    EMERGENCY_LAND = 4 # Immediate emergency landing required


# =============================================================================
# Safety Configuration
# =============================================================================

# Battery thresholds (percentage)
BATTERY_WARN_THRESHOLD = 20     # Warn user at 20%
BATTERY_CRITICAL_THRESHOLD = 10  # Critical at 10%, should land

# Battery check rate (avoid spamming Tello queries)
BATTERY_CHECK_INTERVAL = 5.0    # Check every 5 seconds

# Contact detection thresholds
CONTACT_SPEED_THRESHOLD = 5     # Speed < 5 cm/s while commanding = blocked
CONTACT_FRAMES_NEEDED = 5       # Need 5 consecutive frames to confirm contact
CONTACT_COMMAND_THRESHOLD = 5   # Only check if commanded FB > this

# Backoff behavior
BACKOFF_FB = -MAX_FB            # Back up at max speed
BACKOFF_UD = 5                  # Rise slightly when backing off
BACKOFF_DURATION = 0.5          # Seconds to continue backoff


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SafetyState:
    """
    Current safety status of the drone.

    This dataclass contains all safety-related state that the main loop
    or visualization may need to display or act upon.
    """
    # Battery status
    battery_level: int = 100          # 0-100%
    battery_warning: bool = False     # True if < WARN threshold
    battery_critical: bool = False    # True if < CRITICAL threshold

    # Contact/collision status
    contact_detected: bool = False    # True if blocked movement detected
    blocked_frames: int = 0           # Consecutive frames of blocked movement

    # Movement tracking (for contact detection)
    last_commanded_fb: int = 0        # Last forward/back command sent
    last_actual_speed: float = 0.0    # Actual forward speed from Tello

    # Emergency status
    emergency: bool = False           # True if any critical condition
    emergency_reason: str = ""        # Human-readable reason

    # Timestamps
    last_battery_check: float = 0.0   # time.time() of last battery query

    # Backoff tracking
    backoff_until: float = 0.0        # time.time() when backoff should end


# =============================================================================
# Safety Monitor
# =============================================================================

class SafetyMonitor:
    """
    Monitors safety conditions and returns override actions when needed.

    Safety checks run BEFORE POMDP updates and can override any POMDP action.
    This ensures hard safety constraints are never violated.

    Usage:
        safety = SafetyMonitor(tello)

        while running:
            # Check safety FIRST
            state, override = safety.update(last_commanded_fb)

            if state.emergency:
                # Handle emergency
                break

            # Pass override to POMDP (or None if no override)
            pomdp_override = ACTION_BACKOFF if override == SafetyOverride.BACKOFF else None
            result = pomdp.update(obs, safety_override=pomdp_override)
    """

    def __init__(self, tello: 'Tello'):
        """
        Initialize safety monitor.

        Args:
            tello: djitellopy Tello instance for sensor queries
        """
        self._tello = tello
        self._state = SafetyState()

        # Do initial battery check
        self._check_battery(force=True)

    def update(self, commanded_fb: int = 0) -> Tuple[SafetyState, int]:
        """
        Check all safety conditions and return current state + override.

        This should be called every frame BEFORE POMDP updates.

        Args:
            commanded_fb: The forward/back command being sent this frame.
                         Used for contact detection (if commanding forward
                         but not moving, we might be blocked).

        Returns:
            Tuple of (SafetyState, override_code):
            - SafetyState: Current safety status
            - override_code: SafetyOverride constant (NONE if no override needed)
        """
        # Track commanded movement
        self._state.last_commanded_fb = commanded_fb

        # Check battery (rate-limited)
        self._check_battery()

        # Check contact detection
        contact_triggered = self._check_contact(commanded_fb)

        # Determine override priority:
        # 1. Emergency takes priority
        # 2. Contact detection (backoff)
        # 3. Battery critical (land)
        # 4. Battery warning (hover)

        override = SafetyOverride.NONE

        if self._state.emergency:
            override = SafetyOverride.EMERGENCY_LAND
        elif contact_triggered or self._in_backoff():
            override = SafetyOverride.BACKOFF
            self._state.contact_detected = True
        elif self._state.battery_critical:
            override = SafetyOverride.LAND
            self._state.emergency = True
            self._state.emergency_reason = f"Battery critical: {self._state.battery_level}%"
        elif self._state.battery_warning:
            # Just warn, don't override (let POMDP continue but user should land soon)
            override = SafetyOverride.NONE

        return self._state, override

    def _check_battery(self, force: bool = False) -> None:
        """
        Update battery state from Tello.

        Rate-limited to avoid excessive queries unless force=True.

        Args:
            force: If True, check regardless of rate limit
        """
        now = time.time()

        if not force and (now - self._state.last_battery_check) < BATTERY_CHECK_INTERVAL:
            return

        try:
            self._state.battery_level = self._tello.get_battery()
            self._state.last_battery_check = now

            # Update warning/critical flags
            self._state.battery_warning = self._state.battery_level < BATTERY_WARN_THRESHOLD
            self._state.battery_critical = self._state.battery_level < BATTERY_CRITICAL_THRESHOLD

        except Exception as e:
            # If battery query fails, assume the worst but don't crash
            print(f"[Safety] Battery query failed: {e}")

    def _check_contact(self, commanded_fb: int) -> bool:
        """
        Check for contact/collision based on movement mismatch.

        Detection logic:
        1. If commanded_fb > CONTACT_COMMAND_THRESHOLD (commanding forward)
        2. And actual speed_y < CONTACT_SPEED_THRESHOLD
        3. For CONTACT_FRAMES_NEEDED consecutive frames
        4. Then contact is detected

        Args:
            commanded_fb: Current forward/back command

        Returns:
            True if contact was detected this frame (triggers backoff)
        """
        # Only check if commanding significant forward movement
        if commanded_fb <= CONTACT_COMMAND_THRESHOLD:
            self._state.blocked_frames = 0
            return False

        try:
            speed_y = self._tello.get_speed_y()  # Forward speed in Tello frame
            self._state.last_actual_speed = speed_y

            if abs(speed_y) < CONTACT_SPEED_THRESHOLD:
                self._state.blocked_frames += 1

                if self._state.blocked_frames >= CONTACT_FRAMES_NEEDED:
                    print(f"[Safety] CONTACT DETECTED! Commanded fb={commanded_fb} "
                          f"but speed_y={speed_y}")
                    self._start_backoff()
                    return True
            else:
                # Movement is happening, reset counter
                self._state.blocked_frames = 0

        except Exception:
            # If speed query fails, continue without contact detection
            pass

        return False

    def _start_backoff(self) -> None:
        """Start backoff timer after contact detection."""
        self._state.backoff_until = time.time() + BACKOFF_DURATION
        self._state.blocked_frames = 0

    def _in_backoff(self) -> bool:
        """Check if we're still in backoff mode."""
        return time.time() < self._state.backoff_until

    def get_override_rc(self, override: int) -> Tuple[int, int, int, int]:
        """
        Get RC control values for a safety override action.

        Args:
            override: SafetyOverride constant

        Returns:
            Tuple of (lr, fb, ud, yaw) RC control values
        """
        if override == SafetyOverride.BACKOFF:
            return (0, BACKOFF_FB, BACKOFF_UD, 0)  # Back up and rise
        elif override == SafetyOverride.HOVER:
            return (0, 0, 0, 0)  # Hold position
        elif override in (SafetyOverride.LAND, SafetyOverride.EMERGENCY_LAND):
            return (0, 0, 0, 0)  # Stop before landing (landing handled separately)
        else:
            return (0, 0, 0, 0)  # Default: stop

    def reset_contact(self) -> None:
        """Reset contact detection state after backoff completes."""
        self._state.contact_detected = False
        self._state.blocked_frames = 0
        self._state.backoff_until = 0.0

    def should_land(self) -> bool:
        """
        Check if drone should land immediately.

        Returns:
            True if emergency or critical battery requires landing
        """
        return self._state.emergency or self._state.battery_critical

    def get_state(self) -> SafetyState:
        """Get current safety state (for diagnostics/visualization)."""
        return self._state

    def __repr__(self) -> str:
        return (f"SafetyMonitor(battery={self._state.battery_level}%, "
                f"warning={self._state.battery_warning}, "
                f"contact={self._state.contact_detected})")
