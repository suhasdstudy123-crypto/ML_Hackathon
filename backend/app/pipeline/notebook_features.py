"""Reviewed definitions from the supplied notebook; SHA-256 9fa739dc905d93e7d46fa31eeb1fad9d4d4916c337ce0bdfec804dec766b1b5d.
Regenerate with scripts/extract_notebook_pipeline.py; no algorithm edits here.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class TripData:
    """Route-blind array adapter. Deliberately has no reference route field."""
    trip_id: str
    nodes: np.ndarray
    arcs: np.ndarray
    track: np.ndarray

K_CANDIDATES = 5
NOMINAL_SAMPLING_HZ = 1.0
FEATURE_COLUMNS = ['dist_to_edge_m', 'dist_to_u_node_m', 'dist_to_v_node_m', 'bearing_diff_deg', 'bearing_valid', 'time_gap_prev_s', 'speed_implied_mps', 'irregular_gap_flag', 'candidate_rank', 'dist_minus_best_at_point_m', 'num_candidates_at_point', 'edge_length_m', 'u_node_degree', 'v_node_degree', 'reverse_edge_exists']

EARTH_RADIUS_M = 6_371_000.0

def make_local_projector(lon0: float, lat0: float):
    lat0_rad = np.radians(lat0)
    cos_lat0 = np.cos(lat0_rad)

    def project(lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x = np.radians(lon - lon0) * cos_lat0 * EARTH_RADIUS_M
        y = np.radians(lat - lat0) * EARTH_RADIUS_M
        return x, y

    return project

def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Distance from point P to segment AB, clamped to the segment's endpoints."""
    abx, aby = bx - ax, by - ay
    seg_len_sq = abx * abx + aby * aby
    seg_len_sq = np.where(seg_len_sq == 0, 1e-12, seg_len_sq)  # guard zero-length edges
    t = ((px - ax) * abx + (py - ay) * aby) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    closest_x = ax + t_clamped * abx
    closest_y = ay + t_clamped * aby
    return np.hypot(px - closest_x, py - closest_y)

def initial_bearing_deg(lon1, lat1, lon2, lat2):
    """Great-circle initial bearing in degrees [0, 360); NaN if points coincide."""
    lon1, lat1 = np.radians(lon1), np.radians(lat1)
    lon2, lat2 = np.radians(lon2), np.radians(lat2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    coincident = (np.abs(dlon) < 1e-12) & (np.abs(lat2 - lat1) < 1e-12)
    bearing = np.degrees(np.arctan2(x, y))
    bearing = np.where(bearing < 0, bearing + 360.0, bearing)
    return np.where(coincident, np.nan, bearing)

def angular_difference_deg(a, b):
    """Smallest absolute angular difference between two bearings, in [0, 180]."""
    diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0
    return np.minimum(diff, 360.0 - diff)

def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres — used for the speed feature only."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def generate_candidates(trip: TripData, k: int, overfetch_factor: int = 4, min_pool: int = 20):
    nodes, arcs, track = trip.nodes, trip.arcs, trip.track

    # Project using THIS trip's own centroid — never another trip's.
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    project = make_local_projector(lon0, lat0)

    node_x, node_y = project(nodes[:, 0], nodes[:, 1])
    track_x, track_y = project(track[:, 0], track[:, 1])

    u_idx, v_idx = arcs[:, 0], arcs[:, 1]
    ux, uy = node_x[u_idx], node_y[u_idx]
    vx, vy = node_x[v_idx], node_y[v_idx]
    mid_x, mid_y = (ux + vx) / 2.0, (uy + vy) / 2.0

    # Stage 1 — fast, coarse retrieval over edge midpoints.
    tree = cKDTree(np.column_stack([mid_x, mid_y]))
    pool_size = min(max(k * overfetch_factor, min_pool), len(arcs))
    query_points = np.column_stack([track_x, track_y])
    _, pool_edge_ids = tree.query(query_points, k=pool_size)
    if pool_size == 1:
        pool_edge_ids = pool_edge_ids[:, None]

    n_points = len(track)
    px_pool = np.repeat(track_x, pool_size)
    py_pool = np.repeat(track_y, pool_size)
    pool_flat = pool_edge_ids.reshape(-1)
    pool_dist = point_to_segment_distance(
        px_pool, py_pool, ux[pool_flat], uy[pool_flat], vx[pool_flat], vy[pool_flat]
    ).reshape(n_points, pool_size)

    # Stage 2 — re-rank the pool by TRUE point-to-segment distance, keep top k.
    k_eff = min(k, pool_size)
    order = np.argsort(pool_dist, axis=1)[:, :k_eff]
    candidate_edge_id = np.take_along_axis(pool_edge_ids, order, axis=1).reshape(-1)
    dist_to_edge_m = np.take_along_axis(pool_dist, order, axis=1).reshape(-1)

    point_idx = np.repeat(np.arange(n_points), k_eff)
    px = np.repeat(track_x, k_eff)
    py = np.repeat(track_y, k_eff)
    dist_to_u_node_m = np.hypot(px - ux[candidate_edge_id], py - uy[candidate_edge_id])
    dist_to_v_node_m = np.hypot(px - vx[candidate_edge_id], py - vy[candidate_edge_id])

    return {
        "point_idx": point_idx,
        "candidate_edge_id": candidate_edge_id,
        "dist_to_edge_m": dist_to_edge_m,
        "dist_to_u_node_m": dist_to_u_node_m,
        "dist_to_v_node_m": dist_to_v_node_m,
        "_node_x": node_x, "_node_y": node_y,
        "_track_x": track_x, "_track_y": track_y,
    }

def _vehicle_heading_deg(track):
    """Heading of travel at each point, from consecutive raw GPS fixes.
    First point has no predecessor -> NaN (flagged via bearing_valid, not guessed)."""
    n = len(track)
    heading = np.full(n, np.nan)
    if n > 1:
        heading[1:] = initial_bearing_deg(track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1])
    return heading

def _time_gaps_s(track):
    n = len(track)
    gap = np.full(n, np.nan)
    if n > 1:
        gap[1:] = track[1:, 2] - track[:-1, 2]
    return gap

def _implied_speed_mps(track):
    n = len(track)
    speed = np.full(n, np.nan)
    if n > 1:
        dist = haversine_m(track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1])
        gap = track[1:, 2] - track[:-1, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            speed[1:] = np.where(gap > 0, dist / gap, np.nan)
    return speed

def _node_degrees(arcs, n_nodes):
    """How many road segments meet at each node — structural, not an identifier."""
    degree = np.zeros(n_nodes, dtype=int)
    np.add.at(degree, arcs[:, 0], 1)
    np.add.at(degree, arcs[:, 1], 1)
    return degree

def _reverse_edge_lookup(arcs):
    """For every edge (u, v), does the reverse (v, u) also exist in this trip's graph?"""
    edge_set = set(map(tuple, arcs.tolist()))
    return np.array([(v, u) in edge_set for u, v in arcs], dtype=bool)

def build_feature_table(trip: TripData, cand: dict, nominal_sampling_hz: float) -> pd.DataFrame:
    track, arcs, n_nodes = trip.track, trip.arcs, len(trip.nodes)
    point_idx = cand["point_idx"]
    candidate_edge_id = cand["candidate_edge_id"]
    node_x, node_y = cand["_node_x"], cand["_node_y"]

    vehicle_heading = _vehicle_heading_deg(track)
    time_gap = _time_gaps_s(track)
    speed = _implied_speed_mps(track)

    df = pd.DataFrame({
        "point_idx": point_idx,
        "candidate_edge_id": candidate_edge_id,
        "timestamp": track[point_idx, 2],
        "dist_to_edge_m": cand["dist_to_edge_m"],
        "dist_to_u_node_m": cand["dist_to_u_node_m"],
        "dist_to_v_node_m": cand["dist_to_v_node_m"],
    })

    # --- directional ---
    u_idx = arcs[candidate_edge_id, 0]
    v_idx = arcs[candidate_edge_id, 1]
    edge_heading = initial_bearing_deg(
        trip.nodes[u_idx, 0], trip.nodes[u_idx, 1], trip.nodes[v_idx, 0], trip.nodes[v_idx, 1]
    )
    veh_heading_per_row = vehicle_heading[point_idx]
    bearing_diff = angular_difference_deg(veh_heading_per_row, edge_heading)
    df["bearing_valid"] = (~np.isnan(veh_heading_per_row)).astype(int)
    df["bearing_diff_deg"] = np.where(df["bearing_valid"] == 1, bearing_diff, np.nan)

    # --- kinematic / temporal ---
    df["time_gap_prev_s"] = time_gap[point_idx]
    df["speed_implied_mps"] = speed[point_idx]
    with np.errstate(invalid="ignore"):
        df["irregular_gap_flag"] = (
            np.abs(df["time_gap_prev_s"] - (1.0 / nominal_sampling_hz)) > 0.25
        ).astype(float)
    df.loc[df["time_gap_prev_s"].isna(), "irregular_gap_flag"] = np.nan

    # --- relative / within-point ranking ---
    df["candidate_rank"] = (
        df.groupby("point_idx")["dist_to_edge_m"].rank(method="first").astype(int) - 1
    )
    df["dist_minus_best_at_point_m"] = df["dist_to_edge_m"] - df.groupby("point_idx")[
        "dist_to_edge_m"
    ].transform("min")
    df["num_candidates_at_point"] = df.groupby("point_idx")["candidate_edge_id"].transform("count")

    # --- local topology (structural, generalises across trips) ---
    degree = _node_degrees(arcs, n_nodes)
    reverse_exists = _reverse_edge_lookup(arcs)
    edge_length = np.hypot(node_x[v_idx] - node_x[u_idx], node_y[v_idx] - node_y[u_idx])

    df["edge_length_m"] = edge_length
    df["u_node_degree"] = degree[u_idx]
    df["v_node_degree"] = degree[v_idx]
    df["reverse_edge_exists"] = reverse_exists[candidate_edge_id].astype(int)

    return df

def build_trip_table(trip: TripData, k: int = K_CANDIDATES) -> pd.DataFrame:
    """Route-blind candidate/feature generation; safe for live inference."""
    cand = generate_candidates(trip, k=k)
    df = build_feature_table(trip, cand, nominal_sampling_hz=NOMINAL_SAMPLING_HZ)
    df.insert(0, "trip_id", trip.trip_id)
    return df

