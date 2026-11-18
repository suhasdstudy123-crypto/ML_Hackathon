import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\admin\Downloads\Student folder\Student folder"
TRAIN_DIR = os.path.join(DATA_DIR, "train")

# Let's inspect trip 00000001
trip_dir = os.path.join(TRAIN_DIR, "00000001")
df_track = pd.read_csv(os.path.join(trip_dir, "00000001.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
df_nodes = pd.read_csv(os.path.join(trip_dir, "00000001.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
df_arcs = pd.read_csv(os.path.join(trip_dir, "00000001.arcs"), sep=r'\s+', header=None, names=['u', 'v'])
df_route = pd.read_csv(os.path.join(trip_dir, "00000001.route"), sep=r'\s+', header=None, names=['arc_id'])

print(f"Track shape: {df_track.shape}")
print(f"Nodes shape: {df_nodes.shape}")
print(f"Arcs shape: {df_arcs.shape}")
print(f"Route shape: {df_route.shape}")

print("\nFirst 10 route arcs:")
print(df_route.head(10))

# Check route connectivity: does arc[i].v == arc[i+1].u?
route_arcs = df_route['arc_id'].values
connected = 0
total_transitions = len(route_arcs) - 1
for i in range(total_transitions):
    a1 = route_arcs[i]
    a2 = route_arcs[i+1]
    u1, v1 = df_arcs.iloc[a1]['u'], df_arcs.iloc[a1]['v']
    u2, v2 = df_arcs.iloc[a2]['u'], df_arcs.iloc[a2]['v']
    if v1 == u2:
        connected += 1
    elif a1 == a2:
        # Same arc repeated?
        pass

print(f"\nRoute sequential transitions: {connected} / {total_transitions} are directly connected (v1 == u2)")
if total_transitions > 0:
    for i in range(min(10, total_transitions)):
        a1, a2 = route_arcs[i], route_arcs[i+1]
        print(f"step {i}: arc {a1} ({df_arcs.iloc[a1]['u']} -> {df_arcs.iloc[a1]['v']}) to arc {a2} ({df_arcs.iloc[a2]['u']} -> {df_arcs.iloc[a2]['v']})")
