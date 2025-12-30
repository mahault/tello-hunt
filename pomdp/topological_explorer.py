# pomdp/topological_explorer.py
"""
Topological exploration policy with anti-oscillation and escape behaviors.

Key fixes:
1. ANTI-OSCILLATION: Penalize switching to opposite action
2. BACKOFF ESCAPE: Actually back up when stuck, don't just rotate
3. HEADING-AWARE BLOCKING: Forward blocked at yaw=0 shouldn't poison yaw=90
4. TIME-DECAYED PENALTIES: Blocked penalty decays if not attempted recently
"""

import math
import random
from typing import Optional, Dict

# Action indices matching the simulator
ROTATE_LEFT = 3
ROTATE_RIGHT = 4
FORWARD = 1
BACKWARD = 2
STRAFE_LEFT = 5
STRAFE_RIGHT = 6

ROTATE_ACTIONS = [ROTATE_LEFT, ROTATE_RIGHT]
PRIMARY_ACTIONS = [FORWARD, ROTATE_LEFT, ROTATE_RIGHT]
ESCAPE_MOVES = [BACKWARD, STRAFE_LEFT, STRAFE_RIGHT]

# Opposite actions for anti-oscillation
OPPOSITE_ACTION = {
    ROTATE_LEFT: ROTATE_RIGHT,
    ROTATE_RIGHT: ROTATE_LEFT,
    STRAFE_LEFT: STRAFE_RIGHT,
    STRAFE_RIGHT: STRAFE_LEFT,
    FORWARD: BACKWARD,
    BACKWARD: FORWARD,
}


class TopologicalExplorer:
    """
    Policy for exploring using the topological map.

    Strategy:
    - Normal mode: forward + rotate with anti-oscillation
    - Escape mode: ACTUAL backoff (reverse/strafe) when forward dominated
    - Heading-aware: blocked memory decays with heading change
    """

    def __init__(
        self,
        stuck_window: int = 30,
        blocked_penalty: float = 2.0,
        exploration_bonus: float = 1.5,
        revisit_penalty: float = 0.3,
        oscillation_penalty: float = 1.5,  # Penalty for opposite action
    ):
        self.stuck_window = stuck_window
        self.blocked_penalty = blocked_penalty
        self.exploration_bonus = exploration_bonus
        self.revisit_penalty = revisit_penalty
        self.oscillation_penalty = oscillation_penalty

        # History tracking
        self.last_places = []
        self.stuck_counter = 0

        # Anti-oscillation state
        self.last_action: Optional[int] = None
        self.action_history = []  # Last N actions for oscillation detection
        self.action_history_size = 10

        # Escape mode state
        self.escape_mode = False
        self.escape_phase = 0  # 0=backup, 1=rotate, 2=forward
        self.escape_steps = 0
        self.escape_direction = ROTATE_LEFT

        # Forward block tracking (heading-aware)
        self.forward_blocked_count = 0
        self.last_forward_heading: Optional[float] = None  # Heading when forward was blocked
        self.heading_change_threshold = 0.4  # ~23 degrees - reset blocked if heading changed

        # Movement success tracking
        self.recent_moves = []
        self.move_history_size = 10

        # Room tracking
        self.last_room = None
        self.frames_since_room_change = 0

        # Current heading (updated from pipeline)
        self.current_heading: float = 0.0

    def update_history(self, place_id):
        """Update place history for stuck detection."""
        self.last_places.append(place_id)
        if len(self.last_places) > self.stuck_window:
            self.last_places.pop(0)

    def update_room(self, current_room: str):
        """Track room changes."""
        if self.last_room is not None and current_room != self.last_room:
            self.frames_since_room_change = 0
        else:
            self.frames_since_room_change += 1
        self.last_room = current_room

    def update_heading(self, heading: float):
        """Update current heading (radians)."""
        self.current_heading = heading

    def is_stuck(self) -> bool:
        """Check if stuck (oscillating between same places)."""
        if len(self.last_places) < self.stuck_window:
            return False
        return len(set(self.last_places)) <= 3

    def _is_oscillating(self) -> bool:
        """Detect left-right-left-right oscillation pattern."""
        if len(self.action_history) < 6:
            return False
        # Check for alternating pattern in last 6 actions
        recent = self.action_history[-6:]
        alternating = 0
        for i in range(len(recent) - 1):
            if recent[i] == OPPOSITE_ACTION.get(recent[i+1]):
                alternating += 1
        return alternating >= 4  # 4+ alternations in 6 actions = oscillating

    def record_block(self, was_blocked: bool):
        """Record whether the last movement action was blocked."""
        self.recent_moves.append(not was_blocked)
        if len(self.recent_moves) > self.move_history_size:
            self.recent_moves.pop(0)

        if was_blocked:
            self.forward_blocked_count += 1
            self.last_forward_heading = self.current_heading
        else:
            self.forward_blocked_count = 0
            # Successfully moved - exit escape mode
            if self.escape_mode:
                self.escape_mode = False
                self.escape_phase = 0
                self.escape_steps = 0

    def _should_reset_blocked(self) -> bool:
        """Check if blocked penalty should reset due to heading change."""
        if self.last_forward_heading is None:
            return False
        heading_diff = abs(self.current_heading - self.last_forward_heading)
        # Normalize to [-pi, pi]
        while heading_diff > math.pi:
            heading_diff -= 2 * math.pi
        heading_diff = abs(heading_diff)
        return heading_diff >= self.heading_change_threshold

    def _get_move_success_rate(self) -> float:
        """Get recent movement success rate."""
        if not self.recent_moves:
            return 1.0
        return sum(self.recent_moves) / len(self.recent_moves)

    def choose_action(self, topo_map, place_id: int, debug: bool = False) -> int:
        """
        Choose the best action at the current place.

        Key behaviors:
        1. Anti-oscillation: penalize opposite of last action
        2. Escape mode: actual backoff sequence when stuck
        3. Heading-aware: reset blocked penalty if heading changed
        """
        node = topo_map.get_place(place_id)
        action_names = ['stay', 'fwd', 'back', 'left', 'right', 'str_L', 'str_R']

        # Check if stuck
        stuck = self.is_stuck()
        if stuck:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # Check for oscillation
        oscillating = self._is_oscillating()

        # Reset forward blocked if heading changed significantly
        if self._should_reset_blocked():
            if debug:
                print(f"    [TOPO] Heading changed, resetting forward blocked count")
            self.forward_blocked_count = 0
            self.last_forward_heading = None

        # === ESCAPE MODE ===
        # Enter escape when: stuck OR forward blocked 3+ times OR oscillating
        should_escape = (
            not self.escape_mode and
            (self.stuck_counter > 8 or self.forward_blocked_count > 2 or oscillating)
        )

        if should_escape:
            self.escape_mode = True
            self.escape_phase = 0  # Start with backup
            self.escape_steps = 0
            self.escape_direction = random.choice([ROTATE_LEFT, ROTATE_RIGHT])
            if debug:
                reason = "stuck" if self.stuck_counter > 8 else ("oscillating" if oscillating else "forward blocked")
                print(f"    [TOPO] Entering ESCAPE MODE ({reason})")

        if self.escape_mode:
            self.escape_steps += 1

            # Escape sequence:
            # Phase 0: Back up 2-3 steps
            # Phase 1: Rotate 4-6 steps
            # Phase 2: Try forward
            # Then repeat or exit

            if self.escape_phase == 0:
                # Backup phase
                if self.escape_steps <= 3:
                    if debug:
                        print(f"    [TOPO] ESCAPE: Backing up ({self.escape_steps}/3)")
                    self._record_action(BACKWARD)
                    return BACKWARD
                else:
                    self.escape_phase = 1
                    self.escape_steps = 0

            if self.escape_phase == 1:
                # Rotate phase
                if self.escape_steps <= 5:
                    if debug:
                        print(f"    [TOPO] ESCAPE: Rotating ({self.escape_steps}/5)")
                    self._record_action(self.escape_direction)
                    return self.escape_direction
                else:
                    self.escape_phase = 2
                    self.escape_steps = 0

            if self.escape_phase == 2:
                # Try forward
                if self.escape_steps <= 2:
                    if debug:
                        print(f"    [TOPO] ESCAPE: Trying forward ({self.escape_steps}/2)")
                    self._record_action(FORWARD)
                    return FORWARD
                else:
                    # Full escape cycle done - either exit or repeat
                    if self.forward_blocked_count > 0:
                        # Still blocked, try again with different direction
                        self.escape_phase = 0
                        self.escape_steps = 0
                        self.escape_direction = OPPOSITE_ACTION.get(self.escape_direction, ROTATE_LEFT)
                        if debug:
                            print(f"    [TOPO] ESCAPE: Still blocked, retrying other direction")
                    else:
                        # Success! Exit escape mode
                        self.escape_mode = False
                        self.escape_phase = 0
                        self.escape_steps = 0
                        if debug:
                            print(f"    [TOPO] ESCAPE: Success, returning to normal mode")

        # === NORMAL MODE ===
        action_scores = {}

        for action in PRIMARY_ACTIONS:
            edge = node.edges[action]
            attempts = edge.attempts + 1
            blocked_rate = edge.blocked_rate

            # UCB exploration bonus
            explore = self.exploration_bonus * math.sqrt(
                math.log(node.visits + 1) / attempts
            )

            # Blocked penalty (with heading-aware decay)
            # If heading changed significantly, reduce penalty
            effective_blocked_rate = blocked_rate
            if action == FORWARD and self._should_reset_blocked():
                effective_blocked_rate *= 0.3  # Reduce penalty after heading change

            penalty = self.blocked_penalty * effective_blocked_rate

            # Revisit penalty (superlinear)
            revisit_pen = self.revisit_penalty * (edge.attempts ** 1.2) / 10.0

            score = explore - penalty - revisit_pen

            # Bonus for never-tried actions
            if edge.attempts == 0:
                score += 2.5

            # Bonus for actions leading to new places
            n_destinations = len(edge.next_places)
            if n_destinations > 0:
                score += 0.3 * n_destinations

            # ANTI-OSCILLATION: penalize opposite of last action
            if self.last_action is not None:
                opposite = OPPOSITE_ACTION.get(self.last_action)
                if action == opposite:
                    score -= self.oscillation_penalty
                    if debug:
                        pass  # Will show in score output

            # Room-change bonus
            if self.frames_since_room_change > 20 and action in ROTATE_ACTIONS:
                score += 0.5

            # Prefer forward slightly
            if action == FORWARD:
                score += 0.3

            # Small random factor
            score += random.random() * 0.1

            action_scores[action] = {
                'score': score,
                'attempts': edge.attempts,
                'blocked': edge.blocked,
                'blocked_rate': blocked_rate,
                'explore_bonus': explore,
                'penalty': penalty,
                'revisit_penalty': revisit_pen,
                'osc_penalty': self.oscillation_penalty if (self.last_action and action == OPPOSITE_ACTION.get(self.last_action)) else 0,
                'n_destinations': n_destinations,
            }

        # Find best action
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a]['score'])

        if debug:
            unique_places = len(set(self.last_places[-self.stuck_window:])) if self.last_places else 0
            move_rate = self._get_move_success_rate()
            osc_str = " OSCILLATING!" if oscillating else ""
            print(f"    [TOPO] Place {place_id} (visits={node.visits}, "
                  f"unique={unique_places}, move_rate={move_rate:.0%}{osc_str})")
            for action in sorted(action_scores.keys()):
                info = action_scores[action]
                marker = " <--" if action == best_action else ""
                osc_marker = " [OSC]" if info['osc_penalty'] > 0 else ""
                print(f"      {action_names[action]:6s}: score={info['score']:+.2f} "
                      f"(explore={info['explore_bonus']:.2f}, blocked={info['penalty']:.2f}, "
                      f"revisit={info['revisit_penalty']:.2f}{osc_marker}, "
                      f"attempts={info['attempts']}, dests={info['n_destinations']}){marker}")

        self._record_action(best_action)
        return best_action

    def _record_action(self, action: int):
        """Record action for anti-oscillation tracking."""
        self.last_action = action
        self.action_history.append(action)
        if len(self.action_history) > self.action_history_size:
            self.action_history.pop(0)

    def reset(self):
        """Reset explorer state."""
        self.last_places = []
        self.stuck_counter = 0
        self.last_action = None
        self.action_history = []
        self.escape_mode = False
        self.escape_phase = 0
        self.escape_steps = 0
        self.forward_blocked_count = 0
        self.last_forward_heading = None
        self.recent_moves = []
        self.last_room = None
        self.frames_since_room_change = 0
        self.current_heading = 0.0
