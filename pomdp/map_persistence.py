"""
Map persistence for saving and loading learned topological maps.

Maps are stored in JSON format for human readability and easy debugging.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .topological_map import TopologicalMap


# Default directory for saved maps
DEFAULT_MAPS_DIR = Path(__file__).parent.parent / "maps"


def ensure_maps_dir(maps_dir: Path = DEFAULT_MAPS_DIR) -> Path:
    """Ensure the maps directory exists."""
    maps_dir.mkdir(parents=True, exist_ok=True)
    return maps_dir


def save_map(
    topo_map: TopologicalMap,
    name: str = None,
    maps_dir: Path = DEFAULT_MAPS_DIR,
    metadata: Dict[str, Any] = None
) -> Path:
    """
    Save a topological map to JSON file.

    Args:
        topo_map: The topological map to save
        name: Optional name for the map file (default: timestamp)
        maps_dir: Directory to save maps in
        metadata: Optional metadata to include (e.g., environment name)

    Returns:
        Path to the saved file
    """
    ensure_maps_dir(maps_dir)

    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"learned_map_{timestamp}"

    # Ensure .json extension
    if not name.endswith('.json'):
        name += '.json'

    filepath = maps_dir / name

    # Build save data
    save_data = {
        'version': '1.0',
        'saved_at': datetime.now().isoformat(),
        'n_locations': topo_map.n_locations,
        'n_edges': len(topo_map.edges),
        'map_data': topo_map.to_dict(),
    }

    if metadata:
        save_data['metadata'] = metadata

    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"Map saved to: {filepath}")
    return filepath


def load_map(
    filepath: Path = None,
    name: str = None,
    maps_dir: Path = DEFAULT_MAPS_DIR
) -> Optional[TopologicalMap]:
    """
    Load a topological map from JSON file.

    Args:
        filepath: Direct path to the map file
        name: Name of the map file (looked up in maps_dir)
        maps_dir: Directory containing maps

    Returns:
        Loaded TopologicalMap, or None if not found
    """
    if filepath is None:
        if name is None:
            raise ValueError("Must provide either filepath or name")
        if not name.endswith('.json'):
            name += '.json'
        filepath = maps_dir / name

    filepath = Path(filepath)

    if not filepath.exists():
        print(f"Map file not found: {filepath}")
        return None

    try:
        with open(filepath, 'r') as f:
            save_data = json.load(f)

        topo_map = TopologicalMap.from_dict(save_data['map_data'])

        print(f"Loaded map from: {filepath}")
        print(f"  Locations: {topo_map.n_locations}")
        print(f"  Edges: {len(topo_map.edges)}")

        if 'metadata' in save_data:
            print(f"  Metadata: {save_data['metadata']}")

        return topo_map

    except Exception as e:
        print(f"Error loading map: {e}")
        return None


def load_latest_map(maps_dir: Path = DEFAULT_MAPS_DIR) -> Optional[TopologicalMap]:
    """
    Load the most recently saved map.

    Returns:
        Most recent TopologicalMap, or None if no maps exist
    """
    ensure_maps_dir(maps_dir)

    map_files = list(maps_dir.glob("*.json"))
    if not map_files:
        print("No saved maps found")
        return None

    # Sort by modification time, newest first
    map_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return load_map(filepath=map_files[0])


def list_saved_maps(maps_dir: Path = DEFAULT_MAPS_DIR) -> list:
    """
    List all saved maps with basic info.

    Returns:
        List of dicts with map info
    """
    ensure_maps_dir(maps_dir)

    map_files = list(maps_dir.glob("*.json"))
    map_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    maps_info = []
    for filepath in map_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            maps_info.append({
                'name': filepath.stem,
                'filepath': str(filepath),
                'n_locations': data.get('n_locations', '?'),
                'n_edges': data.get('n_edges', '?'),
                'saved_at': data.get('saved_at', '?'),
                'metadata': data.get('metadata', {}),
            })
        except Exception as e:
            maps_info.append({
                'name': filepath.stem,
                'filepath': str(filepath),
                'error': str(e),
            })

    return maps_info


def delete_map(
    filepath: Path = None,
    name: str = None,
    maps_dir: Path = DEFAULT_MAPS_DIR
) -> bool:
    """
    Delete a saved map file.

    Args:
        filepath: Direct path to the map file
        name: Name of the map file (looked up in maps_dir)
        maps_dir: Directory containing maps

    Returns:
        True if deleted, False if not found
    """
    if filepath is None:
        if name is None:
            raise ValueError("Must provide either filepath or name")
        if not name.endswith('.json'):
            name += '.json'
        filepath = maps_dir / name

    filepath = Path(filepath)

    if filepath.exists():
        filepath.unlink()
        print(f"Deleted map: {filepath}")
        return True
    else:
        print(f"Map not found: {filepath}")
        return False


def export_map_summary(
    topo_map: TopologicalMap,
    filepath: Path = None
) -> str:
    """
    Export a human-readable summary of the map.

    Useful for debugging and understanding learned environments.

    Returns:
        Summary text
    """
    lines = [
        "=" * 60,
        "TOPOLOGICAL MAP SUMMARY",
        "=" * 60,
        f"Total locations: {topo_map.n_locations}",
        f"Total edges: {len(topo_map.edges)}",
        f"Current location: {topo_map.current_location_id}",
        "",
        "LOCATIONS:",
        "-" * 40,
    ]

    for node in topo_map.nodes:
        # Find most prominent objects at this location
        sig = node.observation_signature
        top_objs = []
        sorted_idx = sig.argsort()[::-1]
        for idx in sorted_idx[:3]:  # Top 3
            if sig[idx] > 0.1:
                from .config import TYPE_NAMES
                top_objs.append(f"{TYPE_NAMES[idx]}:{sig[idx]:.2f}")

        neighbors = topo_map.get_neighbors(node.id)
        established = "established" if node.is_established() else "new"

        lines.append(
            f"  [{node.id}] visits={node.visit_count} ({established})"
        )
        if top_objs:
            lines.append(f"       objects: {', '.join(top_objs)}")
        if neighbors:
            neighbor_str = ", ".join([f"{n[0]}(a{n[1]})" for n in neighbors])
            lines.append(f"       neighbors: {neighbor_str}")

    lines.extend([
        "",
        "EDGES:",
        "-" * 40,
    ])

    for edge in topo_map.edges:
        lines.append(
            f"  {edge.from_node_id} --[action {edge.action}]--> "
            f"{edge.to_node_id} (count={edge.count})"
        )

    lines.append("=" * 60)

    summary = "\n".join(lines)

    if filepath:
        with open(filepath, 'w') as f:
            f.write(summary)
        print(f"Summary exported to: {filepath}")

    return summary
