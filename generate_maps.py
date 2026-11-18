"""
PathSnap Map Visualizer Using Folium
Generates complete interactive and screenshot-ready map visualizations:
- Blue circles: Noisy GPS observations
- Green line: Ground truth reference route
- Orange/Red dashed line: Predicted route (HMM Viterbi)
- Purple lines: Local road network geometry
"""

import os
import folium
import numpy as np
import pandas as pd
import joblib
from feature_engineering import load_trip_data, get_projection_scale
from hmm_viterbi import viterbi_decode_trip

DATA_DIR = r"C:\Users\admin\Downloads\Student folder\Student folder"
TEST_DIR = os.path.join(DATA_DIR, "student_test")
MODELS_DIR = r"C:\Users\admin\.gemini\antigravity\scratch\PathSnap\models"
MAPS_DIR = r"C:\Users\admin\.gemini\antigravity\scratch\PathSnap\maps"
os.makedirs(MAPS_DIR, exist_ok=True)

def generate_trip_map(trip_id="00000007"):
    tdir = os.path.join(TEST_DIR, trip_id)
    df_tr, df_nd, df_ar, df_rt = load_trip_data(tdir, trip_id)
    
    # Load tuned model
    model = joblib.load(os.path.join(MODELS_DIR, "best_tuned_histgradientboosting.joblib"))
    
    print(f"Generating map for Trip {trip_id}...")
    clf_arcs, hmm_arcs = viterbi_decode_trip(df_tr, df_nd, df_ar, model)
    
    # Nodes coordinates
    nodes = df_nd[['lon', 'lat']].values
    u_idx = df_ar['u'].values
    v_idx = df_ar['v'].values
    
    center_lat = float(df_tr['lat'].mean())
    center_lon = float(df_tr['lon'].mean())
    
    # Create folium map with standard OpenStreetMap tiles
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')
    
    # 1. Road Network Layer (surrounding arcs within bbox + buffer)
    fg_network = folium.FeatureGroup(name="Surrounding Road Network (Arcs)")
    min_lon, max_lon = df_tr['lon'].min() - 0.01, df_tr['lon'].max() + 0.01
    min_lat, max_lat = df_tr['lat'].min() - 0.01, df_tr['lat'].max() + 0.01
    
    # Sample road network in bbox
    u_lons, u_lats = nodes[u_idx, 0], nodes[u_idx, 1]
    in_bbox = (u_lons >= min_lon) & (u_lons <= max_lon) & (u_lats >= min_lat) & (u_lats <= max_lat)
    bbox_arc_ids = np.where(in_bbox)[0]
    
    # Draw sample road arcs
    for a in bbox_arc_ids[:1500]:
        u, v = u_idx[a], v_idx[a]
        folium.PolyLine(
            locations=[[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
            color='#95a5a6',
            weight=1.5,
            opacity=0.5
        ).add_to(fg_network)
    fg_network.add_to(fmap)
    
    # 2. Ground Truth Route Layer
    fg_gt = folium.FeatureGroup(name="Ground-Truth Route (Green)")
    route_arcs = df_rt['arc_id'].values
    for a in route_arcs:
        u, v = u_idx[a], v_idx[a]
        folium.PolyLine(
            locations=[[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
            color='#27ae60',
            weight=6,
            opacity=0.85,
            tooltip=f"GT Arc {a}"
        ).add_to(fg_gt)
    fg_gt.add_to(fmap)
    
    # 3. Predicted HMM Route Layer
    fg_pred = folium.FeatureGroup(name="Predicted Route (HMM Viterbi - Orange)")
    unique_pred_arcs = pd.Series(hmm_arcs).drop_duplicates().tolist()
    for a in unique_pred_arcs:
        if a >= 0:
            u, v = u_idx[a], v_idx[a]
            folium.PolyLine(
                locations=[[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
                color='#e67e22',
                weight=3.5,
                opacity=0.9,
                dash_array='6',
                tooltip=f"Pred Arc {a}"
            ).add_to(fg_pred)
    fg_pred.add_to(fmap)
    
    # 4. GPS Observations Layer
    fg_gps = folium.FeatureGroup(name="Observed GPS Trajectory (Blue Dots)")
    for i, row in df_tr.iloc[::2].iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3.5,
            color='#2980b9',
            fill=True,
            fill_color='#3498db',
            fill_opacity=0.85,
            popup=f"GPS #{i} (t={row['time']}s)"
        ).add_to(fg_gps)
    fg_gps.add_to(fmap)
    
    # Add Layer Control
    folium.LayerControl(collapsed=False).add_to(fmap)
    
    # Save HTML
    out_html = os.path.join(MAPS_DIR, f"trip_{trip_id}_reconstructed.html")
    fmap.save(out_html)
    print(f"Map successfully saved to {out_html}")
    return out_html

if __name__ == "__main__":
    generate_trip_map("00000007")
    generate_trip_map("00000000")
