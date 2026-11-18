import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

trip_dir = r"C:\Users\admin\Downloads\Student folder\Student folder\train\00000001"
df_track = pd.read_csv(os.path.join(trip_dir, "00000001.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
df_nodes = pd.read_csv(os.path.join(trip_dir, "00000001.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
df_arcs = pd.read_csv(os.path.join(trip_dir, "00000001.arcs"), sep=r'\s+', header=None, names=['u', 'v'])
df_route = pd.read_csv(os.path.join(trip_dir, "00000001.route"), sep=r'\s+', header=None, names=['arc_id'])

nodes = df_nodes[['lon', 'lat']].values
route_arcs = df_route['arc_id'].values

lat_0 = np.radians(df_track['lat'].mean())
scale_x = 111139.0 * np.cos(lat_0)
scale_y = 111139.0
scale = np.array([scale_x, scale_y])

# Calculate ground-truth arc for each GPS observation along route polyline
u_idx_rt = df_arcs.iloc[route_arcs]['u'].values
v_idx_rt = df_arcs.iloc[route_arcs]['v'].values
u_rt_m = nodes[u_idx_rt] * scale
v_rt_m = nodes[v_idx_rt] * scale
tr_m = df_track[['lon', 'lat']].values * scale

gt_arcs = []
for p in tr_m:
    ab = v_rt_m - u_rt_m
    ap = p - u_rt_m
    ab_len_sq = np.sum(ab**2, axis=1)
    ab_len_sq[ab_len_sq == 0] = 1e-6
    t = np.clip(np.sum(ap * ab, axis=1) / ab_len_sq, 0.0, 1.0)
    proj = u_rt_m + t[:, None] * ab
    dist = np.sqrt(np.sum((p - proj)**2, axis=1))
    gt_arcs.append(route_arcs[np.argmin(dist)])
gt_arcs = np.array(gt_arcs)

# Now test candidate generation on all arcs in the map
u_idx_all = df_arcs['u'].values
v_idx_all = df_arcs['v'].values
u_all_m = nodes[u_idx_all] * scale
v_all_m = nodes[v_idx_all] * scale

# Sample 5 points along each arc (0, 0.25, 0.5, 0.75, 1.0)
fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
pts = u_all_m[:, None, :] + fractions[None, :, None] * (v_all_m - u_all_m)[:, None, :]
arc_ids = np.repeat(np.arange(len(u_all_m)), len(fractions))
flat_pts = pts.reshape(-1, 2)

print(f"Total road arcs: {len(u_all_m)}, Total indexed sample points: {len(flat_pts)}")
tree = cKDTree(flat_pts)

for k_query in [20, 40, 60]:
    dists, indices = tree.query(tr_m, k=k_query)
    hits = 0
    for i in range(len(tr_m)):
        cand_arcs = np.unique(arc_ids[indices[i]])
        if gt_arcs[i] in cand_arcs:
            hits += 1
    recall = hits / len(tr_m) * 100
    print(f"Query k={k_query:2d} sample points -> Top candidates recall: {hits}/{len(tr_m)} ({recall:.2f}%), Miss Rate: {100-recall:.2f}%")
