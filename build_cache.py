"""
Dataset Feature Extractor for PathSnap
Extracts and caches tabular candidate feature matrices for all training and student-testing trips.
Uses GroupKFold across trip IDs to prevent data leakage.
"""
import os
import time
import numpy as np
import pandas as pd
from feature_engineering import load_trip_data, extract_features_for_trip

DATA_DIR = r"C:\Users\admin\Downloads\Student folder\Student folder"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "student_test")
CACHE_DIR = r"C:\Users\admin\.gemini\antigravity\scratch\PathSnap\data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def build_dataset_cache(split='train', max_trips=None):
    base_dir = TRAIN_DIR if split == 'train' else TEST_DIR
    trip_ids = sorted(os.listdir(base_dir))
    if max_trips:
        trip_ids = trip_ids[:max_trips]
        
    print(f"Starting feature extraction for {len(trip_ids)} trips in split '{split}'...")
    all_X = []
    all_y = []
    all_groups = [] # trip index for GroupKFold
    all_meta = [] # (trip_id, obs_idx, arc_id)
    
    t0 = time.time()
    for idx, tid in enumerate(trip_ids):
        tdir = os.path.join(base_dir, tid)
        try:
            df_tr, df_nd, df_ar, df_rt = load_trip_data(tdir, tid)
            X_trip, y_trip, cand_info, _ = extract_features_for_trip(df_tr, df_nd, df_ar, df_rt)
            
            all_X.append(X_trip)
            all_y.append(y_trip)
            all_groups.extend([idx] * len(X_trip))
            
            for obs_idx, arc_id in cand_info:
                all_meta.append((tid, obs_idx, arc_id))
                
            if (idx + 1) % 10 == 0 or (idx + 1) == len(trip_ids):
                elapsed = time.time() - t0
                print(f"Processed {idx + 1}/{len(trip_ids)} trips ({elapsed:.1f}s)")
        except Exception as e:
            print(f"Error processing trip {tid}: {e}")
            
    df_X = pd.concat(all_X, ignore_index=True)
    y_arr = np.concatenate(all_y)
    groups_arr = np.array(all_groups)
    df_meta = pd.DataFrame(all_meta, columns=['trip_id', 'obs_idx', 'arc_id'])
    
    # Save cache using npz and csv
    np.savez_compressed(
        os.path.join(CACHE_DIR, f"{split}_data.npz"),
        X=df_X.values,
        y=y_arr,
        groups=groups_arr
    )
    df_meta.to_csv(os.path.join(CACHE_DIR, f"{split}_meta.csv"), index=False)
    with open(os.path.join(CACHE_DIR, f"{split}_feature_names.txt"), "w") as f:
        f.write("\n".join(df_X.columns.tolist()))
    
    print(f"Done! Cached {split}: X={df_X.shape}, y={y_arr.shape}, total time={time.time()-t0:.1f}s")
    return df_X, y_arr, groups_arr, df_meta

if __name__ == "__main__":
    build_dataset_cache('train')
    build_dataset_cache('student_test')
