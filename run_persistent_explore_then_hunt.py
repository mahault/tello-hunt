"""
Persistent Explore-Then-Hunt Runner.

Creates a FullPipelineSimulator, loads saved map + odometry if it exists,
runs EXPLORATION for N frames, saves state, switches to HUNTING for M frames,
saves state again.

On subsequent runs, it resumes from the saved map and does a short "top-up
exploration" focused on frontiers adjacent to unknown, then hunts again.

Usage:
    # First run (no saved state yet)
    python run_persistent_explore_then_hunt.py --frontier --explore_frames 800 --hunt_frames 800

    # Subsequent runs (auto-loads state)
    python run_persistent_explore_then_hunt.py --frontier --resume_explore_frames 250 --hunt_frames 800

    # Stop hunting early when target detected
    python run_persistent_explore_then_hunt.py --stop_on_detect
"""

import os
import time
import argparse
import numpy as np

# Import the simulator class defined in test_full_pipeline.py
from test_full_pipeline import FullPipelineSimulator


def _pack_places_dict(d):
    """dict[int, (int,int)] -> (ids, coords[N,2])"""
    if not d:
        return np.array([], dtype=np.int32), np.zeros((0, 2), dtype=np.int32)
    ids = np.array(list(d.keys()), dtype=np.int32)
    coords = np.array([d[i] for i in ids], dtype=np.int32)
    return ids, coords


def _unpack_places_dict(ids, coords):
    """(ids, coords[N,2]) -> dict[int,(int,int)]"""
    out = {}
    for i, (x, y) in zip(ids.tolist(), coords.tolist()):
        out[int(i)] = (int(x), int(y))
    return out


def _pack_labels_dict(d):
    """dict[int,str] -> (ids, labels[str])"""
    if not d:
        return np.array([], dtype=np.int32), np.array([], dtype=object)
    ids = np.array(list(d.keys()), dtype=np.int32)
    labels = np.array([d[i] for i in ids], dtype=object)
    return ids, labels


def _unpack_labels_dict(ids, labels):
    out = {}
    for i, s in zip(ids.tolist(), labels.tolist()):
        out[int(i)] = str(s)
    return out


def save_pipeline_state(pipeline: FullPipelineSimulator, path: str):
    """
    Persist only what you actually need to resume:
    - occupancy grid + visit counts + trajectory + place markers/labels
    - odometry pose + odometry trajectory (so map aligns on resume)
    - frontier room tracker (optional but useful)
    - blocked-edge memory + frontier blacklist
    - human search POMDP sighting prior
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    occ = pipeline.spatial_mapper.map
    odo = pipeline.spatial_mapper.odometry

    places_ids, places_xy = _pack_places_dict(occ.places)
    label_ids, label_vals = _pack_labels_dict(occ.place_labels)

    # Frontier room tracking (only present in --frontier mode)
    rt = None
    if getattr(pipeline, "use_frontier", False) and hasattr(pipeline, "frontier_explorer"):
        rt = pipeline.frontier_explorer.room_tracker

    # Frontier persistence (blocked edges + target blacklist)
    # Store remaining TTL (expiry - current_frame) so it survives restart
    fe_blocked_edges = np.array([], dtype=object)
    fe_blocked_targets = np.array([], dtype=object)
    fe_frame = np.int32(0)

    if getattr(pipeline, "use_frontier", False) and hasattr(pipeline, "frontier_explorer"):
        fe = pipeline.frontier_explorer
        fe_frame = np.int32(getattr(fe, "_frame_count", 0))

        # Store remaining TTL (expiry - current_frame)
        edges = []
        for (from_c, to_c), expiry in getattr(fe, "_blocked_edges", {}).items():
            remaining = int(expiry) - int(fe_frame)
            if remaining > 0:
                edges.append((tuple(from_c), tuple(to_c), remaining))
        fe_blocked_edges = np.array(edges, dtype=object)

        targets = []
        for cell, expiry in getattr(fe, "_blocked_targets", []):
            remaining = int(expiry) - int(fe_frame)
            if remaining > 0:
                targets.append((tuple(cell), remaining))
        fe_blocked_targets = np.array(targets, dtype=object)

    # Human search POMDP state (sighting prior)
    hs_state = None
    if hasattr(pipeline, "human_search") and pipeline.human_search is not None:
        hs_state = pipeline.human_search.save_state()

    payload = {
        # Map core
        "grid": occ.grid,
        "visit_count": occ.visit_count,
        "trajectory": np.array(occ.trajectory, dtype=np.int32) if occ.trajectory else np.zeros((0, 2), dtype=np.int32),
        "places_ids": places_ids,
        "places_xy": places_xy,
        "label_ids": label_ids,
        "label_vals": label_vals,

        # Map config (so you can validate compatibility)
        "map_width": np.int32(occ.config.width),
        "map_height": np.int32(occ.config.height),
        "map_resolution": np.float32(occ.config.resolution),
        "map_origin_x": np.int32(occ.config.origin_x),
        "map_origin_y": np.int32(occ.config.origin_y),

        # Odometry
        "pose_x": np.float32(odo.pose.x),
        "pose_y": np.float32(odo.pose.y),
        "pose_yaw": np.float32(odo.pose.yaw),
        "odo_traj": np.array(getattr(odo, "trajectory", []), dtype=np.float32),

        # Frontier room transitions (optional)
        "room_current": np.array([rt.current_room], dtype=object) if rt else np.array([""], dtype=object),
        "room_visited": np.array(sorted(list(rt.rooms_visited)), dtype=object) if rt else np.array([], dtype=object),
        "room_transition_count": np.int32(rt.transition_count) if rt else np.int32(0),
        "room_transitions": np.array(
            [(t.from_room, t.to_room, int(t.frame), float(t.position[0]), float(t.position[1])) for t in rt.transitions],
            dtype=object
        ) if rt else np.array([], dtype=object),

        # Frontier blocked-edge memory + blacklist (relative TTL)
        "fe_frame": fe_frame,
        "fe_blocked_edges": fe_blocked_edges,
        "fe_blocked_targets": fe_blocked_targets,

        # Human search POMDP state
        "human_search_state": np.array([hs_state], dtype=object),
    }

    np.savez_compressed(path, **payload)
    print(f"[STATE] Saved to {path}")


def load_pipeline_state(pipeline: FullPipelineSimulator, path: str) -> bool:
    if not os.path.exists(path):
        return False

    data = np.load(path, allow_pickle=True)

    # Basic compatibility check (avoid silently loading mismatched map sizes)
    occ = pipeline.spatial_mapper.map
    if int(data["map_width"]) != occ.config.width or int(data["map_height"]) != occ.config.height:
        raise RuntimeError(
            f"Saved map size {int(data['map_width'])}x{int(data['map_height'])} "
            f"does not match current {occ.config.width}x{occ.config.height}"
        )

    # Restore occupancy map
    occ.grid[:, :] = data["grid"]
    occ.visit_count[:, :] = data["visit_count"]

    traj = data["trajectory"]
    occ.trajectory = [tuple(x) for x in traj.tolist()] if traj.size else []

    occ.places = _unpack_places_dict(data["places_ids"], data["places_xy"])
    occ.place_labels = _unpack_labels_dict(data["label_ids"], data["label_vals"])

    # Restore odometry
    odo = pipeline.spatial_mapper.odometry
    odo.pose.x = float(data["pose_x"])
    odo.pose.y = float(data["pose_y"])
    odo.pose.yaw = float(data["pose_yaw"])

    odo_traj = data["odo_traj"]
    if hasattr(odo, "trajectory"):
        odo.trajectory = [tuple(x) for x in odo_traj.tolist()] if odo_traj.size else [(0.0, 0.0)]

    # Restore frontier room tracker (optional)
    if getattr(pipeline, "use_frontier", False) and hasattr(pipeline, "frontier_explorer"):
        rt = pipeline.frontier_explorer.room_tracker
        rt.current_room = str(data["room_current"][0]) if data["room_current"].size else "Unknown"
        rt.rooms_visited = set([str(x) for x in data["room_visited"].tolist()])
        rt.transition_count = int(data["room_transition_count"])

        rt.transitions = []
        for row in data["room_transitions"].tolist():
            # row = (from_room, to_room, frame, x, y)
            from_room, to_room, frame, x, y = row
            # Import here to avoid circular imports
            from pomdp.frontier_explorer import RoomTransition
            rt.transitions.append(RoomTransition(
                from_room=str(from_room),
                to_room=str(to_room),
                frame=int(frame),
                position=(float(x), float(y))
            ))

    # Restore frontier blocked-edge memory + blacklist (relative TTL)
    if getattr(pipeline, "use_frontier", False) and hasattr(pipeline, "frontier_explorer"):
        fe = pipeline.frontier_explorer
        fe._blocked_edges = {}
        fe._blocked_targets = []
        fe._frame_count = 0  # restart counting; expiries are rebuilt relative to this

        for row in data.get("fe_blocked_edges", np.array([], dtype=object)).tolist():
            from_c, to_c, remaining = row
            fe._blocked_edges[(tuple(from_c), tuple(to_c))] = fe._frame_count + int(remaining)

        for row in data.get("fe_blocked_targets", np.array([], dtype=object)).tolist():
            cell, remaining = row
            fe._blocked_targets.append((tuple(cell), fe._frame_count + int(remaining)))

        n_edges = len(fe._blocked_edges)
        n_targets = len(fe._blocked_targets)
        if n_edges > 0 or n_targets > 0:
            print(f"[STATE] Restored {n_edges} blocked edges, {n_targets} blacklisted targets")

    # Restore HumanSearchPOMDP state (sighting prior)
    hs_arr = data.get("human_search_state", np.array([None], dtype=object))
    hs_state = hs_arr[0] if hs_arr.size else None
    if hs_state is not None:
        from pomdp.human_search import HumanSearchPOMDP
        pipeline.human_search = HumanSearchPOMDP.load_state(hs_state)
        stats = pipeline.human_search.get_statistics()
        print(f"[STATE] Restored HumanSearchPOMDP: {stats['total_sightings']:.0f} sightings, "
              f"last at loc {stats['last_sighting_location']}")

    print(f"[STATE] Loaded from {path}")
    return True


def run_for_frames(pipeline, n_frames, sleep_s=0.0, stop_if=None):
    """
    Run pipeline.update() n_frames times.
    stop_if(result) -> True to break early.
    """
    last = None
    for _ in range(n_frames):
        last = pipeline.update(manual_action=0)
        if stop_if is not None and stop_if(last):
            break
        if sleep_s > 0:
            time.sleep(sleep_s)
    return last


def main():
    ap = argparse.ArgumentParser(description="Persistent explore-then-hunt runner")
    ap.add_argument("--state", default="maps/persistent_state.npz", help="Where to save/load mapping state")
    ap.add_argument("--explore_frames", type=int, default=600, help="Frames to explore before hunting (first run)")
    ap.add_argument("--resume_explore_frames", type=int, default=250, help="Frames to explore on resume (top-up)")
    ap.add_argument("--hunt_frames", type=int, default=600, help="Frames to hunt after exploration")
    ap.add_argument("--frontier", action="store_true", default=True, help="Use frontier exploration (default on)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep per frame (sim debugging)")
    ap.add_argument("--stop_on_detect", action="store_true", help="Stop hunting early when a person/cat is detected")
    args = ap.parse_args()

    # Build pipeline
    pipeline = FullPipelineSimulator(use_semantic=False, use_topological=False, use_lookahead=False, use_frontier=args.frontier)
    pipeline.autonomous = True

    # Try to load previous state
    resumed = load_pipeline_state(pipeline, args.state)

    # ---- Stage 1: Exploration ----
    pipeline.mode = "exploration"
    if hasattr(pipeline, "exploration"):
        pipeline.exploration.reset()
    if getattr(pipeline, "use_frontier", False) and hasattr(pipeline, "frontier_explorer"):
        # Reset action-level chase target; keep room_tracker info if we resumed
        pipeline.frontier_explorer.target = None
        pipeline.frontier_explorer.target_cell = None

    # "Explore a bit" logic: on resume, run shorter top-up exploration
    n_explore = args.resume_explore_frames if resumed else args.explore_frames

    # Optional: stop exploration early if we measurably reduce unknown area
    occ = pipeline.spatial_mapper.map
    stats0 = occ.get_stats()
    unknown0 = stats0["unknown_pct"]

    def stop_explore(res):
        stats = occ.get_stats()
        unknown = stats["unknown_pct"]
        # Stop when unknown % drops by ~1% (tweak to taste)
        return (unknown0 - unknown) >= 1.0

    print(f"[RUN] {'Resuming' if resumed else 'Fresh'} exploration for up to {n_explore} frames...")
    run_for_frames(pipeline, n_explore, sleep_s=args.sleep, stop_if=stop_explore)

    # Save after exploration
    save_pipeline_state(pipeline, args.state)

    # ---- Stage 2: Hunting ----
    pipeline.mode = "hunting"
    if hasattr(pipeline, "interaction"):
        pipeline.interaction.reset_to_searching()

    def stop_hunt(res):
        if not args.stop_on_detect:
            return False
        return bool(res.get("person_detected", False) or res.get("cat_detected", False))

    print(f"[RUN] Hunting for up to {args.hunt_frames} frames...")
    run_for_frames(pipeline, args.hunt_frames, sleep_s=args.sleep, stop_if=stop_hunt)

    # Save after hunting as well (map continues updating while hunting)
    save_pipeline_state(pipeline, args.state)

    # Final summary
    s = pipeline.spatial_mapper.map.get_stats()
    print("[DONE] Map stats:", s)


if __name__ == "__main__":
    main()
