import os
import time
import numpy as np
import pandas as pd
from feature_engineering import load_trip_data, extract_features_for_trip

DATA_DIR = r"C:\Users\admin\Downloads\Student folder\Student folder"
TRAIN_DIR = os.path.join(DATA_DIR, "train")

# Let's test extraction on trip 00000001
t0 = time.time()
df_tr, df_nd, df_ar, df_rt = load_trip_data(os.path.join(TRAIN_DIR, "00000001"), "00000001")
X, y, cand_info, gt_arcs = extract_features_for_trip(df_tr, df_nd, df_ar, df_rt)
t1 = time.time()

print(f"Extraction took {t1 - t0:.2f} seconds")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}, positive labels: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
print("First 5 candidate feature rows:")
print(X.head())

# Check if closest candidate (is_closest == 1) gives good accuracy as baseline
closest_mask = X['is_closest'] == 1
y_closest = y[closest_mask]
print(f"Simple Closest-Candidate Accuracy: {np.mean(y_closest)*100:.2f}%")
