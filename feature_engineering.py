"""
PathSnap Feature Engineering and Preprocessing Pipeline
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def get_projection_scale(lat_deg):
    lat_rad = np.radians(lat_deg)
    scale_x = 111139.0 * np.cos(lat_rad)
    scale_y = 111139.0
    return np.array([scale_x, scale_y])

def load_trip_data(trip_dir, trip_id):
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    
    df_track = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    df_nodes = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'])
    df_arcs = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['u', 'v'])
    
    df_route = None
    if os.path.exists(route_path):
        df_route = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'])
        
    return df_track, df_nodes, df_arcs, df_route

def derive_ground_truth_labels(df_track, df_nodes, df_arcs, df_route, scale):
    """
    Associates each GPS observation with its ground truth route arc.
    """
    nodes = df_nodes[['lon', 'lat']].values
    route_arcs = df_route['arc_id'].values
    
    u_idx = df_arcs.iloc[route_arcs]['u'].values
    v_idx = df_arcs.iloc[route_arcs]['v'].values
    
    u_m = nodes[u_idx] * scale
    v_m = nodes[v_idx] * scale
    tr_m = df_track[['lon', 'lat']].values * scale
    
    ab = v_m - u_m
    ab_len_sq = np.sum(ab**2, axis=1)
    ab_len_sq[ab_len_sq == 0] = 1e-6
    
    gt_arcs = np.zeros(len(tr_m), dtype=int)
    for i in range(len(tr_m)):
        p = tr_m[i]
        ap = p - u_m
        t = np.clip(np.sum(ap * ab, axis=1) / ab_len_sq, 0.0, 1.0)
        proj = u_m + t[:, None] * ab
        dist_sq = np.sum((p - proj)**2, axis=1)
        gt_arcs[i] = route_arcs[np.argmin(dist_sq)]
        
    return gt_arcs

def extract_features_for_trip(df_track, df_nodes, df_arcs, df_route=None, k_query=45, max_candidates=15):
    """
    Extracts features for candidates of each GPS observation in a trip.
    Returns:
      X: DataFrame of candidate features
      y: binary labels (1 if candidate is ground truth, 0 otherwise) - None if df_route is None
      meta: dict with metadata (trip_id, observation indices, candidate arc_ids)
    """
    scale = get_projection_scale(df_track['lat'].mean())
    nodes = df_nodes[['lon', 'lat']].values
    u_idx = df_arcs['u'].values
    v_idx = df_arcs['v'].values
    
    u_m = nodes[u_idx] * scale
    v_m = nodes[v_idx] * scale
    tr_m = df_track[['lon', 'lat']].values * scale
    
    # 1. Sample 5 points per arc for spatial indexing
    fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    pts = u_m[:, None, :] + fractions[None, :, None] * (v_m - u_m)[:, None, :]
    arc_idx_map = np.repeat(np.arange(len(u_m)), len(fractions))
    flat_pts = pts.reshape(-1, 2)
    
    tree = cKDTree(flat_pts)
    _, pt_indices = tree.query(tr_m, k=k_query)
    
    # Precompute GPS kinematics
    N_pts = len(df_track)
    gps_time = df_track['time'].values
    dt = np.diff(gps_time, prepend=gps_time[0] - 1.0)
    dt[dt <= 0] = 1.0
    
    # GPS velocity vectors & heading
    dx = np.diff(tr_m[:, 0], prepend=tr_m[0, 0])
    dy = np.diff(tr_m[:, 1], prepend=tr_m[0, 1])
    gps_speed = np.sqrt(dx**2 + dy**2) / dt
    gps_heading = np.arctan2(dy, dx) # in radians [-pi, pi]
    # For point 0, use next point if possible
    if N_pts > 1:
        gps_heading[0] = np.arctan2(tr_m[1, 1] - tr_m[0, 1], tr_m[1, 0] - tr_m[0, 0])
        gps_speed[0] = np.sqrt((tr_m[1, 0] - tr_m[0, 0])**2 + (tr_m[1, 1] - tr_m[0, 1])**2) / max(0.1, gps_time[1] - gps_time[0])
        
    # Precompute arc properties
    arc_vec = v_m - u_m
    arc_len = np.sqrt(np.sum(arc_vec**2, axis=1))
    arc_len_safe = np.where(arc_len == 0, 1e-6, arc_len)
    arc_heading = np.arctan2(arc_vec[:, 1], arc_vec[:, 0])
    
    gt_arcs = None
    if df_route is not None:
        gt_arcs = derive_ground_truth_labels(df_track, df_nodes, df_arcs, df_route, scale)
        
    features = []
    labels = []
    cand_info = [] # (obs_idx, arc_id)
    
    for i in range(N_pts):
        cand_arcs = np.unique(arc_idx_map[pt_indices[i]])[:max_candidates]
        p = tr_m[i]
        
        # Exact distance to each candidate segment
        c_u = u_m[cand_arcs]
        c_v = v_m[cand_arcs]
        ab = c_v - c_u
        ap = p - c_u
        ab_sq = np.sum(ab**2, axis=1)
        ab_sq[ab_sq == 0] = 1e-6
        t = np.clip(np.sum(ap * ab, axis=1) / ab_sq, 0.0, 1.0)
        proj = c_u + t[:, None] * ab
        dist = np.sqrt(np.sum((p - proj)**2, axis=1))
        
        # Heading difference
        c_arc_h = arc_heading[cand_arcs]
        d_heading = np.abs(gps_heading[i] - c_arc_h)
        d_heading = np.minimum(d_heading, 2 * np.pi - d_heading) # in [0, pi]
        
        # Arc length
        c_arc_len = arc_len[cand_arcs]
        
        # Rank by distance among candidates for this point
        dist_rank = np.argsort(np.argsort(dist))
        
        for c_idx, arc_id in enumerate(cand_arcs):
            # Features:
            # 1. dist_to_segment
            # 2. log_dist
            # 3. heading_diff (rad)
            # 4. cos_heading_diff
            # 5. projection_ratio t (how far along the segment)
            # 6. segment_length
            # 7. gps_speed
            # 8. time_gap (dt)
            # 9. dist_rank
            # 10. is_closest (binary)
            
            d = dist[c_idx]
            dh = d_heading[c_idx]
            feat = [
                d,
                np.log1p(d),
                dh,
                np.cos(dh),
                t[c_idx],
                c_arc_len[c_idx],
                gps_speed[i],
                dt[i],
                dist_rank[c_idx],
                1.0 if dist_rank[c_idx] == 0 else 0.0
            ]
            features.append(feat)
            cand_info.append((i, arc_id))
            
            if gt_arcs is not None:
                labels.append(1 if arc_id == gt_arcs[i] else 0)
                
    feature_names = [
        'dist_to_segment',
        'log_dist',
        'heading_diff',
        'cos_heading_diff',
        'proj_ratio_t',
        'segment_length',
        'gps_speed',
        'time_gap',
        'dist_rank',
        'is_closest'
    ]
    
    X = pd.DataFrame(features, columns=feature_names)
    y = np.array(labels) if gt_arcs is not None else None
    
    return X, y, cand_info, gt_arcs
