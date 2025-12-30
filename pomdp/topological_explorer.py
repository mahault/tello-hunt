# pomdp/topological_explorer.py
"""
Topological exploration policy.

Key principle: LOOK WHERE YOU GO
- Primarily use forward + rotate (always see where you're going)
- Strafe/backward as ESCAPE moves only when truly stuck
- Scan mode: systematically rotate to find clear paths
"""

import math
import random

# Action indices matching the simulator
ROTATE_LEFT = 3
ROTATE_RIGHT = 4
FORWARD = 1
BACKWARD = 2
STRAFE_LEFT = 5
STRAFE_RIGHT = 6

# Primary actions: forward + rotation (look where you go)
ROTATE_ACTIONS = [ROTATE_LEFT, ROTATE_RIGHT]
PRIMARY_ACTIONS = [FORWARD, ROTATE_LEFT, ROTATE_RIGHT]

# Escape moves: use when truly stuck (blind moves, but collision-checked)
ESCAPE_MOVES = [BACKWARD, STRAFE_LEFT, STRAFE_RIGHT]


class TopologicalExplorer:
    """
    Policy for exploring using the topological map.

    Strategy:
    - Normal mode: forward + rotate (UCB-based selection)
    - Scan mode: systematic rotation to find open paths
    - Escape mode: when scan fails, try strafe/backward to break free
    """

    def __init__(
        self,
        stuck_window: int = 40,
        blocked_penalty: float = 2.0,
        exploration_bonus: float = 1.5,
    ):
        self.stuck_window = stuck_window
        self.blocked_penalty = blocked_penalty
        self.exploration_bonus = exploration_bonus

        # History tracking
        self.last_places = []
        self.stuck_counter = 0

        # Scan-and-move state
        self.scan_mode = False
        self.scan_rotations = 0
        self.scan_direction = ROTATE_LEFT
        self.last_action_was_rotation = False
        self.forward_blocked_count = 0

        # Escape mode state
        self.escape_mode = False
        self.escape_attempts = 0

        # Movement success tracking
        self.recent_moves = []
        self.move_history_size = 10

    def update_history(self, place_id):
        """Update place history for stuck detection."""
        self.last_places.append(place_id)
        if len(self.last_places) > self.stuck_window:
            self.last_places.pop(0)

    def is_stuck(self) -> bool:
        """Check if stuck (oscillating between same places)."""
        if len(self.last_places) < self.stuck_window:
            return False
        return len(set(self.last_places)) <= 3

    def record_block(self, was_blocked: bool):
        """Record whether the last movement action was blocked."""
        self.recent_moves.append(not was_blocked)
        if len(self.recent_moves) > self.move_history_size:
            self.recent_moves.pop(0)

        if was_blocked:
            self.forward_blocked_count += 1
        else:
            self.forward_blocked_count = 0
            # Successfully moved - exit scan/escape mode
            if self.scan_mode:
                self.scan_mode = False
                self.scan_rotations = 0
            if self.escape_mode:
                self.escape_mode = False
                self.escape_attempts = 0

    def _get_move_success_rate(self) -> float:
        """Get recent movement success rate."""
        if not self.recent_moves:
            return 1.0
        return sum(self.recent_moves) / len(self.recent_moves)

    def choose_action(self, topo_map, place_id: int, debug: bool = False) -> int:
        """
        Choose the best action at the current place.

        Strategy:
        1. Normal mode: forward + rotate (UCB-based)
        2. Scan mode: rotate systematically, try forward after each rotation
        3. Escape mode: when scan fails, try strafe/backward
        """
        node = topo_map.get_place(place_id)
        action_names = ['stay', 'fwd', 'back', 'left', 'right', 'str_L', 'str_R']

        # Check if stuck
        stuck = self.is_stuck()
        if stuck:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # === ESCAPE MODE ===
        # When scan mode completed a full rotation without finding a path
        if self.escape_mode:
            MAX_ESCAPE_ATTEMPTS = 6

            if self.escape_attempts >= MAX_ESCAPE_ATTEMPTS:
                # Escape failed, go back to normal mode
                self.escape_mode = False
                self.escape_attempts = 0
                if debug:
                    print(f"    [TOPO-DECISION] Escape failed, returning to normal mode")
            else:
                # Try an escape move (strafe or backward)
                self.escape_attempts += 1
                # Cycle through escape moves
                escape_action = ESCAPE_MOVES[self.escape_attempts % len(ESCAPE_MOVES)]
                if debug:
                    print(f"    [TOPO-DECISION] ESCAPE: Trying {action_names[escape_action]} "
                          f"({self.escape_attempts}/{MAX_ESCAPE_ATTEMPTS})")
                return escape_action

        # === SCAN MODE ===
        # Enter scan mode when stuck or forward keeps getting blocked
        should_scan = (
            not self.scan_mode and
            not self.escape_mode and
            (self.stuck_counter > 10 or self.forward_blocked_count > 2)
        )

        if should_scan:
            self.scan_mode = True
            self.scan_rotations = 0
            self.scan_direction = random.choice([ROTATE_LEFT, ROTATE_RIGHT])
            self.last_action_was_rotation = False
            if debug:
                dir_name = "left" if self.scan_direction == ROTATE_LEFT else "right"
                reason = "stuck" if self.stuck_counter > 10 else "forward blocked"
                print(f"    [TOPO-DECISION] Entering SCAN MODE ({reason}, rotating {dir_name})")

        if self.scan_mode:
            MAX_SCAN_ROTATIONS = 24  # Full circle

            if self.scan_rotations >= MAX_SCAN_ROTATIONS:
                # Full scan complete - enter escape mode
                self.scan_mode = False
                self.scan_rotations = 0
                self.escape_mode = True
                self.escape_attempts = 0
                if debug:
                    print(f"    [TOPO-DECISION] Full scan complete, entering ESCAPE MODE")
                # Return first escape move
                return BACKWARD

            if self.last_action_was_rotation:
                # Just rotated, now try forward
                self.last_action_was_rotation = False
                if debug:
                    print(f"    [TOPO-DECISION] SCAN: Trying forward ({self.scan_rotations}/{MAX_SCAN_ROTATIONS})")
                return FORWARD
            else:
                # Rotate to next direction
                self.last_action_was_rotation = True
                self.scan_rotations += 1
                if debug:
                    dir_name = "left" if self.scan_direction == ROTATE_LEFT else "right"
                    print(f"    [TOPO-DECISION] SCAN: Rotating {dir_name} ({self.scan_rotations}/{MAX_SCAN_ROTATIONS})")
                return self.scan_direction

        # === NORMAL MODE ===
        # Use only forward + rotate (look where you go)
        action_scores = {}

        for action in PRIMARY_ACTIONS:
            edge = node.edges[action]
            attempts = edge.attempts + 1
            blocked_rate = edge.blocked_rate

            # UCB exploration bonus
            explore = self.exploration_bonus * math.sqrt(
                math.log(node.visits + 1) / attempts
            )

            # Penalty for blocked actions
            penalty = self.blocked_penalty * blocked_rate

            score = explore - penalty

            # Bonus for never-tried actions
            if edge.attempts == 0:
                score += 2.0

            # Bonus for actions that lead to NEW places
            n_destinations = len(edge.next_places)
            if n_destinations > 0:
                score += 0.5 * n_destinations

            # Prefer forward movement
            if action == FORWARD:
                score += 0.5

            # Add small random factor
            score += random.random() * 0.1

            action_scores[action] = {
                'score': score,
                'attempts': edge.attempts,
                'blocked': edge.blocked,
                'blocked_rate': blocked_rate,
                'explore_bonus': explore,
                'penalty': penalty,
                'n_destinations': n_destinations,
            }

        # Find best action
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a]['score'])

        if debug:
            unique_places = len(set(self.last_places[-self.stuck_window:])) if self.last_places else 0
            move_rate = self._get_move_success_rate()
            print(f"    [TOPO-DECISION] Place {place_id} (visits={node.visits}, "
                  f"unique_recent={unique_places}, move_rate={move_rate:.0%})")
            for action in sorted(action_scores.keys()):
                info = action_scores[action]
                marker = " <--" if action == best_action else ""
                print(f"      {action_names[action]:6s}: score={info['score']:+.2f} "
                      f"(explore={info['explore_bonus']:.2f}, penalty={info['penalty']:.2f}, "
                      f"attempts={info['attempts']}, blocked={info['blocked']}, "
                      f"dests={info['n_destinations']}){marker}")

        self.last_action_was_rotation = best_action in ROTATE_ACTIONS
        return best_action

    def reset(self):
        """Reset explorer state."""
        self.last_places = []
        self.stuck_counter = 0
        self.scan_mode = False
        self.scan_rotations = 0
        self.last_action_was_rotation = False
        self.forward_blocked_count = 0
        self.escape_mode = False
        self.escape_attempts = 0
        self.recent_moves = []
