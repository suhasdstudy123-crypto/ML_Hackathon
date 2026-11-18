import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Let's inspect ground truth matching between GPS observations and route arcs.
# In map matching:
# Given a trip with N GPS points and M ground truth traversed route arcs (which form a continuous path),
# each GPS point t is matched to an arc in the road network.
# Let's check how ground truth arc for each GPS observation is defined or can be derived.

trip_dir = r"C:\Users\admin\Downloads\Student folder\Student folder\train\00000001"
df_track = pd.read_csv(os.path.join(trip_dir, "00000001.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
df_nodes = pd.read_csv(os.path.join(trip_dir, "00000001.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
df_arcs = pd.read_csv(os.path.join(trip_dir, "00000001.arcs"), sep=r'\s+', header=None, names=['u', 'v'])
df_route = pd.read_csv(os.path.join(trip_dir, "00000001.route"), sep=r'\s+', header=None, names=['arc_id'])

# Nodes
nodes = df_nodes[['lon', 'lat']].values
route_arcs = df_route['arc_id'].values

print(f"Total track points: {len(df_track)}")
print(f"Total route arcs: {len(route_arcs)}")

# Let's check projection of track onto route arcs
# Route forms a sequential polyline
u_idx = df_arcs.iloc[route_arcs]['u'].values
v_idx = df_arcs.iloc[route_arcs]['v'].values

u_pts = nodes[u_idx] # (M, 2)
v_pts = nodes[v_idx] # (M, 2)

# Midpoint projection factor in meters
lat_0 = np.radians(df_track['lat'].mean())
scale_x = 111139.0 * np.cos(lat_0)
scale_y = 111139.0

# Scale to meters
u_m = u_pts * np.array([scale_x, scale_y])
v_m = v_pts * np.array([scale_x, scale_y])
tr_m = df_track[['lon', 'lat']].values * np.array([scale_x, scale_y])

# For each GPS point, find the closest segment along the route
# Since the vehicle moves monotonically along the route, let's find the closest route segment
closest_route_arc = []
min_dists = []
for i in range(len(tr_m)):
    p = tr_m[i]
    # distance to all route segments
    ab = v_m - u_m
    ap = p - u_m
    ab_len_sq = np.sum(ab**2, axis=1)
    ab_len_sq[ab_len_sq == 0] = 1e-6
    t = np.clip(np.sum(ap * ab, axis=1) / ab_len_sq, 0.0, 1.0)
    proj = u_m + t[:, None] * ab
    dist = np.sqrt(np.sum((p - proj)**2, axis=1))
    best_idx = np.argmin(dist)
    closest_route_arc.append(route_arcs[best_idx])
    min_dists.append(dist[best_idx])

min_dists = np.array(min_dists)
print(f"GPS-to-route distance: mean={min_dists.mean():.2f}m, median={np.median(min_dists):.2f}m, 95th={np.percentile(min_dists, 95):.2f}m, max={min_dists.max():.2f}m")

# Check monotonic progression along route
progression = [np.where(route_arcs == a)[0][0] for a in closest_route_arc]
diffs = np.diff(progression)
print(f"Progression steps: forward={np.sum(diffs >= 0)}, backward={np.sum(diffs < 0)}, total={len(diffs)}")
