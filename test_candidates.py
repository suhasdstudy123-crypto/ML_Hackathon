import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Let's test candidate retrieval for trip 00000001
trip_dir = r"C:\Users\admin\Downloads\Student folder\Student folder\train\00000001"
df_track = pd.read_csv(os.path.join(trip_dir, "00000001.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
df_nodes = pd.read_csv(os.path.join(trip_dir, "00000001.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
df_arcs = pd.read_csv(os.path.join(trip_dir, "00000001.arcs"), sep=r'\s+', header=None, names=['u', 'v'])
df_route = pd.read_csv(os.path.join(trip_dir, "00000001.route"), sep=r'\s+', header=None, names=['arc_id'])

nodes = df_nodes[['lon', 'lat']].values
u_idx = df_arcs['u'].values
v_idx = df_arcs['v'].values
u_pts = nodes[u_idx]
v_pts = nodes[v_idx]

# Midpoint of all arcs in network
mid_pts = (u_pts + v_pts) / 2.0

lat_0 = np.radians(df_track['lat'].mean())
scale_x = 111139.0 * np.cos(lat_0)
scale_y = 111139.0
scale = np.array([scale_x, scale_y])

mid_m = mid_pts * scale
tr_m = df_track[['lon', 'lat']].values * scale

# Build KDTree on arc midpoints
tree = cKDTree(mid_m)

# Query k nearest arcs for each GPS point
k_candidates = 15
dists, indices = tree.query(tr_m, k=k_candidates)

print(f"Candidate retrieval for {len(tr_m)} points with k={k_candidates}:")
print("Sample candidates for point 0:", indices[0], "dists (m):", dists[0])

# Now let's calculate exact point-to-segment distance for candidate arcs
# and see if ground-truth arc is in top-k
from inspect_matching import closest_route_arc
gt_arcs = np.array(closest_route_arc)

hits = 0
for i in range(len(tr_m)):
    if gt_arcs[i] in indices[i]:
        hits += 1

print(f"Top-{k_candidates} Candidate Recall: {hits}/{len(tr_m)} ({hits/len(tr_m)*100:.2f}%)")
print(f"Candidate Miss Rate: {(1 - hits/len(tr_m))*100:.2f}%")
