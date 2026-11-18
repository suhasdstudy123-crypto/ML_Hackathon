import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import folium

DATA_DIR = r"C:\Users\admin\Downloads\Student folder\Student folder"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "student_test")
OUTPUT_DIR = r"C:\Users\admin\.gemini\antigravity\scratch\PathSnap\eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_dataset():
    records = []
    
    for split, base_dir in [('train', TRAIN_DIR), ('student_test', TEST_DIR)]:
        trips = sorted(os.listdir(base_dir))
        for trip_id in trips:
            tdir = os.path.join(base_dir, trip_id)
            track_file = os.path.join(tdir, f"{trip_id}.track")
            nodes_file = os.path.join(tdir, f"{trip_id}.nodes")
            arcs_file = os.path.join(tdir, f"{trip_id}.arcs")
            route_file = os.path.join(tdir, f"{trip_id}.route")
            
            # File sizes
            s_track = os.path.getsize(track_file)
            s_nodes = os.path.getsize(nodes_file)
            s_arcs = os.path.getsize(arcs_file)
            s_route = os.path.getsize(route_file) if os.path.exists(route_file) else 0
            
            # Read track
            df_tr = pd.read_csv(track_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
            n_points = len(df_tr)
            
            # Time delta analysis
            dt = df_tr['time'].diff().dropna()
            exact_1s = (dt == 1.0).sum() / max(1, len(dt))
            max_dt = dt.max() if len(dt) > 0 else 0
            min_dt = dt.min() if len(dt) > 0 else 0
            
            # Check missing
            missing_vals = df_tr.isnull().sum().sum()
            
            # Count nodes and arcs
            # fast line count
            with open(nodes_file, 'r') as f:
                n_nodes = sum(1 for _ in f)
            with open(arcs_file, 'r') as f:
                n_arcs = sum(1 for _ in f)
            n_route = 0
            if os.path.exists(route_file):
                with open(route_file, 'r') as f:
                    n_route = sum(1 for _ in f)
                    
            records.append({
                'split': split,
                'trip_id': trip_id,
                'n_points': n_points,
                'n_nodes': n_nodes,
                'n_arcs': n_arcs,
                'n_route': n_route,
                'duration_s': df_tr['time'].iloc[-1] - df_tr['time'].iloc[0],
                'exact_1s_pct': exact_1s * 100,
                'max_gap_s': max_dt,
                'missing_vals': missing_vals,
                'size_track_kb': s_track / 1024,
                'size_nodes_kb': s_nodes / 1024,
                'size_arcs_kb': s_arcs / 1024,
                'lon_min': df_tr['lon'].min(),
                'lon_max': df_tr['lon'].max(),
                'lat_min': df_tr['lat'].min(),
                'lat_max': df_tr['lat'].max(),
            })
            
    df_meta = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "dataset_inventory.csv")
    df_meta.to_csv(csv_path, index=False)
    print(f"Saved dataset inventory to {csv_path}")
    
    # Print summary statistics
    print("\n--- DATASET SUMMARY BY SPLIT ---")
    summary = df_meta.groupby('split').agg({
        'trip_id': 'count',
        'n_points': ['sum', 'mean', 'min', 'max'],
        'n_nodes': ['mean', 'min', 'max'],
        'n_arcs': ['mean', 'min', 'max'],
        'n_route': ['mean', 'min', 'max'],
        'exact_1s_pct': 'mean',
        'max_gap_s': 'max'
    })
    print(summary)
    
    # Save summary table as CSV for Report
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))
    
    # 1. Plot Trajectory Length Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df_meta, x='n_points', hue='split', kde=True, bins=25, palette='viridis')
    plt.title("Distribution of GPS Observation Counts per Trip (Train vs Student Test)")
    plt.xlabel("Number of GPS Observations")
    plt.ylabel("Frequency (Trips)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "trajectory_length_dist.png"), dpi=300)
    plt.close()
    
    # 2. Plot Sampling Interval Analysis
    # Let's aggregate sampling intervals across 10 random trips
    sample_trips = df_meta[df_meta['split']=='train']['trip_id'].iloc[:15]
    all_dts = []
    for tid in sample_trips:
        tf = os.path.join(TRAIN_DIR, tid, f"{tid}.track")
        tr = pd.read_csv(tf, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
        all_dts.extend(tr['time'].diff().dropna().values)
    all_dts = np.array(all_dts)
    
    plt.figure(figsize=(9, 4.5))
    val_counts = pd.Series(all_dts).value_counts().head(5)
    sns.barplot(x=val_counts.index.astype(str), y=val_counts.values, palette='mako')
    plt.title(f"GPS Sampling Interval Distribution (Nominal 1 Hz, 1s = {(all_dts==1.0).mean()*100:.2f}%)")
    plt.xlabel("Time Delta (seconds)")
    plt.ylabel("Observation Count")
    for i, v in enumerate(val_counts.values):
        plt.text(i, v + len(all_dts)*0.01, f"{v} ({v/len(all_dts)*100:.1f}%)", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sampling_interval_dist.png"), dpi=300)
    plt.close()
    
    # 3. Road Network Graph Analysis for a sample trip
    # Plotting road network and GPS trajectory for trip 00000001
    tid = "00000001"
    tdir = os.path.join(TRAIN_DIR, tid)
    df_tr = pd.read_csv(os.path.join(tdir, f"{tid}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    df_nd = pd.read_csv(os.path.join(tdir, f"{tid}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
    df_ar = pd.read_csv(os.path.join(tdir, f"{tid}.arcs"), sep=r'\s+', header=None, names=['u', 'v'])
    df_rt = pd.read_csv(os.path.join(tdir, f"{tid}.route"), sep=r'\s+', header=None, names=['arc_id'])
    
    # Static GPS Trajectory Plot
    plt.figure(figsize=(10, 8))
    plt.plot(df_tr['lon'], df_tr['lat'], 'o-', markersize=2, alpha=0.6, color='blue', label='GPS Track Points')
    # Plot route segments
    route_u = df_ar.iloc[df_rt['arc_id']]['u'].values
    route_v = df_ar.iloc[df_rt['arc_id']]['v'].values
    ux, uy = df_nd.iloc[route_u]['lon'].values, df_nd.iloc[route_u]['lat'].values
    vx, vy = df_nd.iloc[route_v]['lon'].values, df_nd.iloc[route_v]['lat'].values
    
    for k in range(len(route_u)):
        plt.plot([ux[k], vx[k]], [uy[k], vy[k]], color='green', linewidth=2.5, alpha=0.8,
                 label='Ground Truth Route' if k == 0 else "")
                 
    plt.title(f"Trip {tid}: Observed GPS Trajectory vs Ground Truth Road Segments")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "trip_trajectory_comparison.png"), dpi=300)
    plt.close()
    
    # 4. Interactive Folium Map
    center_lat = df_tr['lat'].mean()
    center_lon = df_tr['lon'].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')
    
    # Add GPS Points as FeatureGroup
    fg_gps = folium.FeatureGroup(name="Noisy GPS Observations")
    for idx, row in df_tr.iloc[::2].iterrows(): # subsample every 2 points for map performance
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color='#1f77b4',
            fill=True,
            fill_color='#1f77b4',
            fill_opacity=0.8,
            popup=f"GPS #{idx} (t={row['time']}s)"
        ).add_to(fg_gps)
    fg_gps.add_to(fmap)
    
    # Add Ground Truth Route as FeatureGroup
    fg_gt = folium.FeatureGroup(name="Ground-Truth Route (Road Arcs)")
    for k in range(len(route_u)):
        folium.PolyLine(
            locations=[[uy[k], ux[k]], [vy[k], vx[k]]],
            color='#2ca02c',
            weight=4,
            opacity=0.9,
            popup=f"Arc ID: {df_rt['arc_id'].iloc[k]}"
        ).add_to(fg_gt)
    fg_gt.add_to(fmap)
    
    # Layer control
    folium.LayerControl().add_to(fmap)
    map_html_path = os.path.join(OUTPUT_DIR, "trip_00000001_folium.html")
    fmap.save(map_html_path)
    print(f"Interactive Folium map saved to {map_html_path}")
    print("EDA completed successfully!")

if __name__ == "__main__":
    analyze_dataset()
