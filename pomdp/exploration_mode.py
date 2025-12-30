"""
Topological Frontier-Based Exploration.

Uses a graph of places and connections for systematic exploration:
1. Places = ORB keyframe nodes (from CSCG place recognition)
2. Edges = Connections discovered when moving between places
3. Frontier = Places with untried movement directions

Exploration strategy:
- If current place has untried directions, try one
- Otherwise, backtrack to nearest frontier node (BFS)

This approach naturally handles:
- Finding doors (untried directions that lead to new places)
- Not getting stuck (always have a frontier to explore)
- Efficient coverage (BFS ensures we explore nearby frontiers first)
"""

from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Set, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .world_model import WorldModel, LocalizationResult
    from .observation_encoder import ObservationToken


# Movement actions
MOVEMENT_ACTIONS = ['stay', 'forward', 'backward', 'left', 'right']
N_ACTIONS = 5

# Actions that can discover new places (translation + rotation)
EXPLORATORY_ACTIONS = [1, 2, 3, 4]  # forward, backward, left, right

# Legacy state constants for backwards compatibility
SCANNING = 0
APPROACHING_FRONTIER = 1
BACKTRACKING = 2
TRANSITIONING = 3


# =============================================================================
# Exploration Graph
# =============================================================================

@dataclass
class ExplorationGraph:
    """
    Topological graph for frontier-based exploration.

    Tracks:
    - Known places (nodes)
    - Edges between places (which action leads where)
    - Which actions have been tried from each place
    - Blocked actions (walls) at each place
    """

    # edges[place_id][action] = destination_place_id (or None if blocked)
    edges: Dict[int, Dict[int, Optional[int]]] = field(default_factory=lambda: defaultdict(dict))

    # tried[place_id] = set of actions we've attempted from this place
    tried: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))

    # blocked[place_id] = set of actions that hit walls
    blocked: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))

    # All known places
    places: Set[int] = field(default_factory=set)

    # For backtracking: parent[place_id] = (parent_place_id, action_to_get_here)
    parent: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    def add_place(self, place_id: int) -> None:
        """Register a new place."""
        self.places.add(place_id)

    def record_transition(self, from_place: int, action: int, to_place: int) -> None:
        """Record a successful transition between places."""
        self.add_place(from_place)
        self.add_place(to_place)
        self.edges[from_place][action] = to_place
        self.tried[from_place].add(action)

        # Record parent for backtracking (if not already set)
        if to_place not in self.parent and to_place != from_place:
            self.parent[to_place] = (from_place, action)

    def record_blocked(self, place_id: int, action: int) -> None:
        """Record that an action is blocked (wall) at a place."""
        self.add_place(place_id)
        self.blocked[place_id].add(action)
        self.tried[place_id].add(action)

    def record_tried(self, place_id: int, action: int) -> None:
        """Record that we tried an action (regardless of outcome)."""
        self.add_place(place_id)
        self.tried[place_id].add(action)

    def get_untried_actions(self, place_id: int) -> List[int]:
        """Get actions not yet tried from this place."""
        tried = self.tried.get(place_id, set())
        return [a for a in EXPLORATORY_ACTIONS if a not in tried]

    def get_unblocked_untried(self, place_id: int) -> List[int]:
        """Get untried actions that aren't known to be blocked."""
        tried = self.tried.get(place_id, set())
        blocked = self.blocked.get(place_id, set())
        return [a for a in EXPLORATORY_ACTIONS
                if a not in tried and a not in blocked]

    def is_frontier(self, place_id: int) -> bool:
        """A place is a frontier if it has untried exploratory actions."""
        return len(self.get_untried_actions(place_id)) > 0

    def get_frontier_places(self) -> List[int]:
        """Get all places that are frontiers."""
        return [p for p in self.places if self.is_frontier(p)]

    def find_nearest_frontier(self, current_place: int) -> Optional[int]:
        """
        BFS to find the nearest frontier place.

        Returns:
            place_id of nearest frontier, or None if no frontiers exist
        """
        if self.is_frontier(current_place):
            return current_place

        visited = {current_place}
        queue = deque([current_place])

        while queue:
            place = queue.popleft()

            # Check all neighbors
            for action, neighbor in self.edges.get(place, {}).items():
                if neighbor is None or neighbor in visited:
                    continue

                if self.is_frontier(neighbor):
                    return neighbor

                visited.add(neighbor)
                queue.append(neighbor)

        return None  # No frontiers reachable

    def get_path_to(self, from_place: int, to_place: int) -> Optional[List[int]]:
        """
        BFS to find path (sequence of actions) from one place to another.

        Returns:
            List of actions to take, or None if no path exists
        """
        if from_place == to_place:
            return []

        visited = {from_place}
        queue = deque([(from_place, [])])

        while queue:
            place, path = queue.popleft()

            for action, neighbor in self.edges.get(place, {}).items():
                if neighbor is None or neighbor in visited:
                    continue

                new_path = path + [action]

                if neighbor == to_place:
                    return new_path

                visited.add(neighbor)
                queue.append((neighbor, new_path))

        return None  # No path found

    def get_stats(self) -> Dict:
        """Get exploration statistics."""
        n_places = len(self.places)
        n_edges = sum(len(e) for e in self.edges.values())
        n_frontiers = len(self.get_frontier_places())
        total_tried = sum(len(t) for t in self.tried.values())
        total_blocked = sum(len(b) for b in self.blocked.values())

        return {
            'n_places': n_places,
            'n_edges': n_edges,
            'n_frontiers': n_frontiers,
            'total_tried': total_tried,
            'total_blocked': total_blocked,
            'exploration_ratio': total_tried / max(1, n_places * len(EXPLORATORY_ACTIONS)),
        }

    def reset(self) -> None:
        """Clear all exploration data."""
        self.edges.clear()
        self.tried.clear()
        self.blocked.clear()
        self.places.clear()
        self.parent.clear()


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class ExplorationResult:
    """Result from exploration mode update."""

    # Selected movement action (0-4: stay, forward, back, left, right)
    selected_action: int
    selected_action_name: str

    # Exploration state
    exploration_state: str  # 'exploring_frontier', 'backtracking', 'complete'

    # Graph stats
    n_places: int
    n_frontiers: int
    n_edges: int

    # Current place info
    current_place: int
    untried_actions: List[int]

    # Backtracking info (if applicable)
    backtrack_target: Optional[int] = None
    backtrack_path: Optional[List[int]] = None

    # Transition recommendation
    should_transition_to_hunt: bool = False
    transition_reason: str = ""

    # For compatibility
    current_vfe: float = 0.0
    mean_vfe: float = 0.0
    vfe_variance: float = 0.0
    locations_discovered: int = 0
    frames_in_exploration: int = 0
    exploration_state_idx: int = 0


# =============================================================================
# Frontier-Based Exploration
# =============================================================================

class ExplorationModePOMDP:
    """
    Topological frontier-based exploration.

    Strategy:
    1. If current place has untried directions, try one (prefer forward)
    2. Otherwise, backtrack to nearest frontier node
    3. When no frontiers remain, exploration is complete
    """

    def __init__(
        self,
        min_places_for_completion: int = 10,
        min_frames: int = 300,
        **kwargs  # Accept legacy params for compatibility
    ):
        """
        Initialize exploration.

        Args:
            min_places_for_completion: Minimum places before considering complete
            min_frames: Minimum frames before considering transition to hunt
        """
        self.min_places = min_places_for_completion
        self.min_frames = min_frames

        # The exploration graph
        self.graph = ExplorationGraph()

        # State tracking
        self._current_place: Optional[int] = None
        self._prev_place: Optional[int] = None
        self._last_action: int = 0
        self._total_frames: int = 0

        # Backtracking state
        self._backtrack_target: Optional[int] = None
        self._backtrack_path: List[int] = []
        self._backtrack_idx: int = 0

        # State name
        self._state: str = 'exploring_frontier'

        # VFE tracking for compatibility
        self._vfe_history: deque = deque(maxlen=100)

    def update(
        self,
        obs: 'ObservationToken',
        world_model: 'WorldModel',
        loc_result: 'LocalizationResult',
        depth_map: Optional[np.ndarray] = None,  # Not used in topological approach
    ) -> ExplorationResult:
        """
        Update exploration state and select action.

        Args:
            obs: Current observation token
            world_model: WorldModel for place info
            loc_result: Localization result with current place
            depth_map: (unused) Depth map for compatibility

        Returns:
            ExplorationResult with selected action
        """
        self._total_frames += 1

        # Track VFE for compatibility
        self._vfe_history.append(loc_result.vfe)

        # Get current place
        current_place = loc_result.token
        if isinstance(current_place, str) and current_place.startswith('Place_'):
            current_place = int(current_place.split('_')[1])

        # Register place
        self.graph.add_place(current_place)

        # Record transition from previous place
        if self._prev_place is not None and self._last_action > 0:
            if current_place != self._prev_place:
                # Successful move to new place - record the edge
                self.graph.record_transition(self._prev_place, self._last_action, current_place)
            # NOTE: Don't mark forward/backward as "blocked" just because place didn't change.
            # The drone might still be moving within a large place.
            # Only record_movement_result() should mark walls (based on simulator feedback).

        # Update state
        self._prev_place = self._current_place
        self._current_place = current_place

        # Select action
        action, state = self._select_action(current_place)
        self._last_action = action
        self._state = state

        # Check for exploration completion
        should_transition, reason = self._check_completion()

        # Build result
        stats = self.graph.get_stats()
        untried = self.graph.get_untried_actions(current_place)

        return ExplorationResult(
            selected_action=action,
            selected_action_name=MOVEMENT_ACTIONS[action],
            exploration_state=state,
            n_places=stats['n_places'],
            n_frontiers=stats['n_frontiers'],
            n_edges=stats['n_edges'],
            current_place=current_place,
            untried_actions=untried,
            backtrack_target=self._backtrack_target,
            backtrack_path=self._backtrack_path if self._backtrack_path else None,
            should_transition_to_hunt=should_transition,
            transition_reason=reason,
            current_vfe=loc_result.vfe,
            mean_vfe=float(np.mean(list(self._vfe_history))) if self._vfe_history else 0.0,
            vfe_variance=float(np.var(list(self._vfe_history))) if self._vfe_history else 0.0,
            locations_discovered=stats['n_places'],
            frames_in_exploration=self._total_frames,
        )

    def _select_action(self, current_place: int) -> Tuple[int, str]:
        """
        Select action using frontier-based exploration.

        Returns:
            (action_idx, state_name)
        """
        debug = (self._total_frames % 30 == 0)

        # If we're backtracking, continue on path
        if self._backtrack_path and self._backtrack_idx < len(self._backtrack_path):
            action = self._backtrack_path[self._backtrack_idx]
            self._backtrack_idx += 1

            # Check if we've reached the target
            if self._backtrack_idx >= len(self._backtrack_path):
                if debug:
                    print(f"  [TOPO] Reached backtrack target {self._backtrack_target}")
                self._backtrack_path = []
                self._backtrack_target = None
                self._backtrack_idx = 0
            else:
                if debug:
                    print(f"  [TOPO] Backtracking step {self._backtrack_idx}/{len(self._backtrack_path)}")

            return action, 'backtracking'

        # Get untried actions at current place
        untried = self.graph.get_untried_actions(current_place)

        if untried:
            # Current place is a frontier - try an untried action
            # Prefer forward > rotations > backward
            priority = [1, 3, 4, 2]  # forward, left, right, backward
            for action in priority:
                if action in untried:
                    # Mark as tried BEFORE returning (so we don't pick it again)
                    self.graph.record_tried(current_place, action)
                    if debug:
                        print(f"  [TOPO] Frontier: trying {MOVEMENT_ACTIONS[action]} "
                              f"({len(untried)-1} untried remaining at place {current_place})")
                    return action, 'exploring_frontier'

            # Fallback: first untried
            action = untried[0]
            self.graph.record_tried(current_place, action)
            if debug:
                print(f"  [TOPO] Frontier: trying {MOVEMENT_ACTIONS[action]}")
            return action, 'exploring_frontier'

        # No untried actions here - need to backtrack to a frontier
        frontier = self.graph.find_nearest_frontier(current_place)

        if frontier is None:
            # No frontiers in the graph - but we might still need to explore!
            stats = self.graph.get_stats()

            if stats['n_places'] < self.min_places:
                # Not enough places discovered - do aggressive random exploration
                # to break out of local area and discover new places

                # Track escape attempts
                if not hasattr(self, '_escape_counter'):
                    self._escape_counter = 0
                self._escape_counter += 1

                # Cycle through: turn, turn, forward, turn, turn, backward, repeat
                # This helps escape corners and find new areas
                escape_sequence = [3, 4, 1, 3, 4, 2, 4, 3, 1, 4, 3, 2]
                action = escape_sequence[self._escape_counter % len(escape_sequence)]

                if debug and self._escape_counter % 10 == 1:
                    print(f"  [TOPO] Stuck with {stats['n_places']} places (need {self.min_places}). "
                          f"Escape sequence step {self._escape_counter}")

                return action, 'escaping'

            # Truly complete - enough places explored
            if debug:
                print(f"  [TOPO] No frontiers remaining - exploration complete! "
                      f"({stats['n_places']} places discovered)")
            return 0, 'complete'  # stay

        # Find path to frontier
        path = self.graph.get_path_to(current_place, frontier)

        if path is None or len(path) == 0:
            # Can't find path - try random exploration
            if debug:
                print(f"  [TOPO] Can't find path to frontier {frontier}, trying random")
            return np.random.choice([1, 3, 4]), 'exploring_frontier'

        # Start backtracking
        self._backtrack_target = frontier
        self._backtrack_path = path
        self._backtrack_idx = 1  # We'll return the first action now

        if debug:
            print(f"  [TOPO] Backtracking to frontier {frontier} via {len(path)} steps")

        return path[0], 'backtracking'

    def _check_completion(self) -> Tuple[bool, str]:
        """Check if exploration should transition to hunting."""
        stats = self.graph.get_stats()

        # Need minimum frames
        if self._total_frames < self.min_frames:
            return False, f"Exploring ({self._total_frames}/{self.min_frames} frames)"

        # Need minimum places
        if stats['n_places'] < self.min_places:
            return False, f"Need more places ({stats['n_places']}/{self.min_places})"

        # Check if no frontiers remain
        if stats['n_frontiers'] == 0:
            return True, f"No frontiers remaining ({stats['n_places']} places explored)"

        # Check exploration ratio
        if stats['exploration_ratio'] > 0.8:
            return True, f"Well explored (ratio={stats['exploration_ratio']:.2f})"

        return False, f"Exploring ({stats['n_frontiers']} frontiers remaining)"

    def record_movement_result(self, action: int, succeeded: bool) -> None:
        """Record whether a movement succeeded (for wall detection)."""
        if self._current_place is not None and not succeeded:
            if action in [1, 2]:  # forward/backward blocked
                self.graph.record_blocked(self._current_place, action)

    def reset(self) -> None:
        """Reset exploration state."""
        self.graph.reset()
        self._current_place = None
        self._prev_place = None
        self._last_action = 0
        self._total_frames = 0
        self._backtrack_target = None
        self._backtrack_path = []
        self._backtrack_idx = 0
        self._state = 'exploring_frontier'
        self._vfe_history.clear()

    def get_vfe_stats(self) -> Tuple[float, float]:
        """Get VFE statistics (for compatibility)."""
        if not self._vfe_history:
            return 0.0, 0.0
        vfe_array = np.array(list(self._vfe_history))
        return float(np.mean(vfe_array)), float(np.var(vfe_array))

    def get_statistics(self) -> dict:
        """Get exploration statistics."""
        stats = self.graph.get_stats()
        return {
            'state': self._state,
            'total_frames': self._total_frames,
            **stats,
        }

    def should_transition_to_hunt(self, world_model) -> Tuple[bool, str]:
        """Check if should transition (for compatibility)."""
        return self._check_completion()

    def __repr__(self) -> str:
        stats = self.graph.get_stats()
        return (f"ExplorationModePOMDP(state={self._state}, "
                f"places={stats['n_places']}, frontiers={stats['n_frontiers']})")


# =============================================================================
# RC Control Conversion (for compatibility)
# =============================================================================

def exploration_action_to_rc_control(
    action_idx: int,
    scan_yaw: int = 40,
    explore_fb: int = 30,
) -> Tuple[int, int, int, int]:
    """
    Convert exploration action to RC control values.

    Args:
        action_idx: Movement action (0-4)
        scan_yaw: Rotation speed
        explore_fb: Forward/backward speed

    Returns:
        (lr, fb, ud, yaw) RC control tuple
    """
    if action_idx == 0:  # stay
        return (0, 0, 0, 0)
    elif action_idx == 1:  # forward
        return (0, explore_fb, 0, 0)
    elif action_idx == 2:  # back
        return (0, -explore_fb, 0, 0)
    elif action_idx == 3:  # left
        return (0, 0, 0, -scan_yaw)
    elif action_idx == 4:  # right
        return (0, 0, 0, scan_yaw)
    return (0, 0, 0, 0)
