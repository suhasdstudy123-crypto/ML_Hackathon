import os
import pandas as pd
import numpy as np

# How are GPS observations associated with the route arcs?
# Let's project GPS points onto the sequence of route segments or test distance to the route.
trip_dir = r"C:\Users\admin\Downloads\Student folder\Student folder\train\00000001"
df_track = pd.read_csv(os.path.join(trip_dir, "00000001.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
df_nodes = pd.read_csv(os.path.join(trip_dir, "00000001.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
df_arcs = pd.read_csv(os.path.join(trip_dir, "00000001.arcs"), sep=r'\s+', header=None, names=['u', 'v'])
df_route = pd.read_csv(os.path.join(trip_dir, "00000001.route"), sep=r'\s+', header=None, names=['arc_id'])

# Nodes coordinates as array
nodes_coords = df_nodes[['lon', 'lat']].values

# Arcs coordinates: (N_arcs, 2, 2)
# u_coords = nodes_coords[df_arcs['u'].values]
# v_coords = nodes_coords[df_arcs['v'].values]

route_arc_ids = df_route['arc_id'].values
u_idx = df_arcs.iloc[route_arc_ids]['u'].values
v_idx = df_arcs.iloc[route_arc_ids]['v'].values

u_coords = nodes_coords[u_idx]
v_coords = nodes_coords[v_idx]

print("First GPS point:", df_track.iloc[0].values)
print("Start route arc 0 u:", u_coords[0], "v:", v_coords[0])
print("Last GPS point:", df_track.iloc[-1].values)
print("End route arc -1 u:", u_coords[-1], "v:", v_coords[-1])

# Point to segment distance in meters (approx lat/lon to meters using local projection)
# For Frankfurt area (lat ~ 50 deg):
# 1 deg lat ~ 111,139 m
# 1 deg lon ~ 111,139 * cos(50) ~ 71,438 m
lat_mid = np.radians(df_track['lat'].mean())
m_per_lat = 111139.0
m_per_lon = 111139.0 * np.cos(lat_mid)

def pt_to_seg_dist(p_x, p_y, a_x, a_y, b_x, b_y):
    # Vector ab
    ab_x = b_x - a_x
    ab_y = b_y - a_y
    ap_x = p_x - a_x
    ap_y = p_y - a_y
    ab_len_sq = ab_x**2 + ab_y**2
    if ab_len_sq == 0:
        return np.sqrt(ap_x**2 + ap_y**2)
    t = np.clip((ap_x * ab_x + ap_y * ab_y) / ab_len_sq, 0.0, 1.0)
    proj_x = a_x + t * ab_x
    proj_y = a_y + t * ab_y
    return np.sqrt((p_x - proj_x)**2 + (p_y - proj_y)**2)

# Check distances of first 5 GPS points to route arc 0, 1, 2...
for i in range(5):
    px = df_track.iloc[i]['lon'] * m_per_lon
    py = df_track.iloc[i]['lat'] * m_per_lat
    # check closest route arc
    dists = []
    for r in range(min(10, len(route_arc_ids))):
        ax = u_coords[r, 0] * m_per_lon
        ay = u_coords[r, 1] * m_per_lat
        bx = v_coords[r, 0] * m_per_lon
        by = v_coords[r, 1] * m_per_lat
        d = pt_to_seg_dist(px, py, ax, ay, bx, by)
        dists.append((d, route_arc_ids[r], r))
    dists.sort()
    print(f"GPS point {i} (t={df_track.iloc[i]['time']}): closest route arc is {dists[0][1]} (index {dists[0][2]}) with dist {dists[0][0]:.2f} m")
