import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\admin\Downloads\Student folder\Student folder"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "student_test")

def inspect_trip(trip_dir, trip_id):
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    
    # Read track: lon, lat, timestamp
    df_track = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    # Read nodes: lon, lat (0-indexed)
    df_nodes = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'])
    # Read arcs: from_node, to_node (0-indexed arc_id)
    df_arcs = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['from_node', 'to_node'])
    
    has_route = os.path.exists(route_path)
    df_route = None
    if has_route:
        df_route = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'])
        
    return {
        'trip_id': trip_id,
        'track_points': len(df_track),
        'nodes_count': len(df_nodes),
        'arcs_count': len(df_arcs),
        'route_arcs': len(df_route) if df_route is not None else 0,
        'time_start': df_track['time'].iloc[0],
        'time_end': df_track['time'].iloc[-1],
        'duration_s': df_track['time'].iloc[-1] - df_track['time'].iloc[0],
        'lon_min': df_track['lon'].min(),
        'lon_max': df_track['lon'].max(),
        'lat_min': df_track['lat'].min(),
        'lat_max': df_track['lat'].max(),
    }

print("Inspecting sample training trip 00000001:")
info = inspect_trip(os.path.join(TRAIN_DIR, "00000001"), "00000001")
for k, v in info.items():
    print(f"  {k}: {v}")

print("\nInspecting sample student test trip 00000000:")
info_test = inspect_trip(os.path.join(TEST_DIR, "00000000"), "00000000")
for k, v in info_test.items():
    print(f"  {k}: {v}")

# Now let's do a survey over all train and test trips
train_trips = sorted(os.listdir(TRAIN_DIR))
test_trips = sorted(os.listdir(TEST_DIR))
print(f"\nTotal train trips: {len(train_trips)}")
print(f"Total student_test trips: {len(test_trips)}")
