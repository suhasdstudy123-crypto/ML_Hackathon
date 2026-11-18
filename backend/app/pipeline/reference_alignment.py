"""Reviewed definitions from the supplied notebook; SHA-256 9fa739dc905d93e7d46fa31eeb1fad9d4d4916c337ce0bdfec804dec766b1b5d.
Regenerate with scripts/extract_notebook_pipeline.py; no algorithm edits here.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .notebook_features import TripData as FeatureTrip, make_local_projector, point_to_segment_distance, _vehicle_heading_deg, _implied_speed_mps, _time_gaps_s, initial_bearing_deg, angular_difference_deg

@dataclass(frozen=True)
class TripData(FeatureTrip):
    """Protected evaluator-only input, constructed after inference completes."""
    route: np.ndarray

ALIGNMENT_SIGMA_M = 15.0
ALIGNMENT_MAX_DISTANCE_M = 70.0
ALIGNMENT_MAX_SPEED_MPS = 55.0
ALIGNMENT_JUMP_SLACK_M = 40.0

def align_gps_to_route(
    trip: TripData,
    sigma_m=ALIGNMENT_SIGMA_M,
    max_distance_m=ALIGNMENT_MAX_DISTANCE_M,
    max_speed_mps=ALIGNMENT_MAX_SPEED_MPS,
    jump_slack_m=ALIGNMENT_JUMP_SLACK_M,
):
    """Estimate a monotonic GPS-to-ordered-route position for labeling only.

    Runs ONLY when the .route reference is present; not used during inference.
    Alignment is estimated/pseudo-ground truth, not independent pointwise truth.
    The complete reference route is used, NOT the five generated candidates.
    """
    if trip.route is None:
        raise ValueError("A reference .route is required for training alignment")
    if sigma_m <= 0 or max_distance_m < 0 or max_speed_mps <= 0:
        raise ValueError("Invalid alignment parameter")

    route = np.asarray(trip.route, dtype=np.int64).reshape(-1)
    track = trip.track
    T, R = len(track), len(route)
    if T == 0 or R == 0:
        raise ValueError(f"{trip.trip_id}: empty GPS track or reference route")
    if np.any(route < 0) or np.any(route >= len(trip.arcs)):
        raise ValueError(f"{trip.trip_id}: reference edge ID outside arc table")

    # Position on the ordered route, NOT a de-duplicated edge ID.
    # This also distinguishes repeated visits to the same edge.
    route_arcs = trip.arcs[route]

    lon0 = float(trip.nodes[:, 0].mean())
    lat0 = float(trip.nodes[:, 1].mean())
    project = make_local_projector(lon0, lat0)
    node_x, node_y = project(trip.nodes[:, 0], trip.nodes[:, 1])
    px, py = project(track[:, 0], track[:, 1])

    ax = node_x[route_arcs[:, 0]]
    ay = node_y[route_arcs[:, 0]]
    bx = node_x[route_arcs[:, 1]]
    by = node_y[route_arcs[:, 1]]

    lengths = np.hypot(bx - ax, by - ay)
    route_start = np.r_[0.0, np.cumsum(lengths[:-1])]
    route_center = route_start + lengths / 2.0
    route_total = float(lengths.sum())

    heading = _vehicle_heading_deg(track)
    speed = _implied_speed_mps(track)
    gaps = _time_gaps_s(track)
    road_heading = initial_bearing_deg(
        trip.nodes[route_arcs[:, 0], 0],
        trip.nodes[route_arcs[:, 0], 1],
        trip.nodes[route_arcs[:, 1], 0],
        trip.nodes[route_arcs[:, 1], 1],
    )

    # O(T*R) time and O(T*R) backpointer memory; vectorized over R.
    # A state is a POSITION in the ordered route, including repeated edges.
    back = np.zeros((T, R), dtype=np.int32)
    ids = np.arange(R, dtype=np.int32)
    previous = None

    for i in range(T):
        if np.isfinite(px[i]) and np.isfinite(py[i]):
            distance = point_to_segment_distance(
                px[i], py[i], ax, ay, bx, by
            )
            # Huber-style geometric penalty; robust against noisy GPS fixes.
            z = np.minimum(distance / sigma_m, 8.0)
            emission = np.where(z <= 2.0, 0.5 * z**2, 2.0 * (z - 1.0))
        else:
            emission = np.zeros(R, dtype=float)

        if (
            np.isfinite(heading[i])
            and np.isfinite(speed[i])
            and speed[i] >= 2.0
            and np.isfinite(gaps[i])
            and 0 < gaps[i] <= 10.0
        ):
            heading_difference = angular_difference_deg(heading[i], road_heading)
            emission += np.nan_to_num(
                0.5 * (heading_difference / 90.0)**2, nan=0.0
            )

        if i == 0:
            # Soft preference for starting near the beginning of the route.
            previous = emission + 0.008 * route_center
            continue

        dt = gaps[i]
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0

        # Forward-only DP. Minimizing prev[k] + w*(pos[j]-pos[k]), k<=j,
        # can be computed with a prefix minimum rather than O(R^2) pairs.
        weight = 0.008 / max(dt, 1.0)
        adjusted = previous - weight * route_center
        prefix_min = np.minimum.accumulate(adjusted)
        is_new_best = adjusted < np.r_[np.inf, prefix_min[:-1]]
        predecessor = np.maximum.accumulate(np.where(is_new_best, ids, 0))
        back[i] = predecessor
        previous = emission + weight * route_center + prefix_min

    # Soft preference to finish near the end of the route.
    final_cost = previous + 0.008 * (route_total - route_center)
    j = int(np.argmin(final_cost))
    positions = np.empty(T, dtype=np.int32)
    for i in range(T - 1, -1, -1):
        positions[i] = j
        if i > 0:
            j = int(back[i, j])

    selected_edges = route[positions]
    distance_to_match = point_to_segment_distance(
        px, py, ax[positions], ay[positions], bx[positions], by[positions]
    )

    # Project each GPS point onto its assigned segment, in route-distance units.
    dx, dy = bx[positions] - ax[positions], by[positions] - ay[positions]
    denominator = np.maximum(dx * dx + dy * dy, 1e-12)
    fraction = np.clip(
        ((px - ax[positions]) * dx + (py - ay[positions]) * dy)
        / denominator, 0.0, 1.0
    )
    along_route_m = route_start[positions] + fraction * lengths[positions]

    # Quality flags: unreliable observations are not given supervised labels.
    status = np.full(T, "accepted", dtype=object)
    invalid_gps = (
        ~np.isfinite(px) | ~np.isfinite(py) | ~np.isfinite(track[:, 2])
    )
    status[invalid_gps] = "invalid_gps"
    status[~invalid_gps & (distance_to_match > max_distance_m)] = (
        "uncertain_distance"
    )

    if T > 1:
        dt = np.diff(track[:, 2])
        route_progress = np.diff(along_route_m)
        bad_time = ~np.isfinite(dt) | (dt <= 0)
        status[np.flatnonzero(bad_time) + 1] = "invalid_timestamp"
        jump = (
            ~bad_time
            & (route_progress > max_speed_mps * dt + jump_slack_m)
        )
        status[np.flatnonzero(jump) + 1] = "uncertain_transition"

        # Gaps between consecutive arcs in the known reference sequence.
        breaks = np.zeros(R, dtype=np.int32)
        if R > 1:
            breaks[1:] = (
                route_arcs[:-1, 1] != route_arcs[1:, 0]
            ).astype(np.int32)
        cumulative_breaks = np.cumsum(breaks)
        crosses_gap = (
            cumulative_breaks[positions[1:]]
            > cumulative_breaks[positions[:-1]]
        )
        status[np.flatnonzero(crosses_gap) + 1] = "uncertain_route_gap"
        # Preserve invalid GPS and invalid timestamp flags as stronger reasons.
        status[invalid_gps] = "invalid_gps"
        status[np.flatnonzero(bad_time) + 1] = "invalid_timestamp"

    return pd.DataFrame({
        "point_idx": np.arange(T, dtype=np.int32),
        "aligned_route_pos": positions,
        "aligned_edge_id": selected_edges,
        "alignment_distance_m": distance_to_match,
        "aligned_route_distance_m": along_route_m,
        "alignment_status": status,
    })
