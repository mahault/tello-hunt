# pomdp/topological_map.py
"""
Pure topological map for camera-only exploration.

Based on place-graph approach: nodes represent distinct visual places,
edges represent traversable transitions learned from experience.
No SLAM, no room semantics required.
"""

from collections import defaultdict
import math


class EdgeStats:
    """Statistics for a (place, action) -> next_place transition."""

    def __init__(self):
        self.attempts = 0
        self.blocked = 0
        self.next_places = defaultdict(int)

    def record(self, moved: bool, next_place):
        """Record a transition attempt."""
        self.attempts += 1
        if not moved:
            self.blocked += 1
        if next_place is not None:
            self.next_places[next_place] += 1

    @property
    def blocked_rate(self):
        """Fraction of attempts that were blocked."""
        if self.attempts == 0:
            return 0.0
        return self.blocked / self.attempts


class PlaceNode:
    """A node in the topological map representing a distinct visual place."""

    def __init__(self, place_id):
        self.place_id = place_id
        self.visits = 0
        self.edges = defaultdict(EdgeStats)  # action -> EdgeStats

    def observe(self):
        """Record a visit to this place."""
        self.visits += 1


class TopologicalMap:
    """
    Learned topological map of the environment.

    The map grows as the drone explores:
    - Places are identified by the place recognizer (ORB keyframes)
    - Edges record which actions lead to which places
    - Blocked actions are tracked to avoid walls
    """

    def __init__(self):
        self.places = {}

    def get_place(self, place_id) -> PlaceNode:
        """Get or create a place node."""
        if place_id not in self.places:
            self.places[place_id] = PlaceNode(place_id)
        return self.places[place_id]

    def observe_place(self, place_id):
        """Record that we observed this place."""
        self.get_place(place_id).observe()

    def record_transition(self, from_place, action, to_place, moved: bool):
        """
        Record a transition from one place to another.

        Args:
            from_place: Place ID where action was taken
            action: Action index that was executed
            to_place: Resulting place ID
            moved: Whether movement succeeded (False if blocked)
        """
        node = self.get_place(from_place)
        node.edges[action].record(moved, to_place)

    def neighbors(self, place_id) -> set:
        """Get all places reachable from this place."""
        node = self.get_place(place_id)
        neigh = set()
        for edge in node.edges.values():
            for p in edge.next_places:
                neigh.add(p)
        return neigh

    def get_stats(self) -> dict:
        """Get map statistics."""
        total_edges = 0
        for place in self.places.values():
            total_edges += len(place.edges)

        return {
            'n_places': len(self.places),
            'n_edges': total_edges,
            'total_visits': sum(p.visits for p in self.places.values()),
        }

    # Compatibility properties for existing code
    @property
    def n_locations(self) -> int:
        """Number of places (alias for len(places))."""
        return len(self.places)

    @property
    def edges(self) -> list:
        """Get all edges as a flat list (for compatibility)."""
        all_edges = []
        for place in self.places.values():
            for action, edge_stats in place.edges.items():
                for next_place, count in edge_stats.next_places.items():
                    all_edges.append({
                        'from': place.place_id,
                        'action': action,
                        'to': next_place,
                        'count': count,
                        'blocked': edge_stats.blocked,
                    })
        return all_edges

    @property
    def nodes(self) -> list:
        """Get all place nodes as a list (for compatibility)."""
        return list(self.places.values())

    @property
    def current_location_id(self) -> int:
        """Current location (not tracked in simple version)."""
        return -1

    def to_dict(self) -> dict:
        """Serialize map to dictionary."""
        places_data = {}
        for pid, place in self.places.items():
            edges_data = {}
            for action, edge_stats in place.edges.items():
                edges_data[str(action)] = {
                    'attempts': edge_stats.attempts,
                    'blocked': edge_stats.blocked,
                    'next_places': dict(edge_stats.next_places),
                }
            places_data[str(pid)] = {
                'place_id': place.place_id,
                'visits': place.visits,
                'edges': edges_data,
            }
        return {'places': places_data}

    @classmethod
    def from_dict(cls, data: dict) -> 'TopologicalMap':
        """Deserialize map from dictionary."""
        topo_map = cls()
        for pid_str, place_data in data.get('places', {}).items():
            pid = int(pid_str)
            place = topo_map.get_place(pid)
            place.visits = place_data.get('visits', 0)
            for action_str, edge_data in place_data.get('edges', {}).items():
                action = int(action_str)
                edge_stats = place.edges[action]
                edge_stats.attempts = edge_data.get('attempts', 0)
                edge_stats.blocked = edge_data.get('blocked', 0)
                for next_p_str, count in edge_data.get('next_places', {}).items():
                    edge_stats.next_places[int(next_p_str)] = count
        return topo_map
