"""
PathSnap FastAPI Server (Laptop 1: Model and API Service)
Hosts the trained machine learning pipeline, HMM/Viterbi route decoder,
and server-side evaluation logic without leaking ground-truth labels to the client.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feature_engineering import load_trip_data, get_projection_scale, derive_ground_truth_labels
from hmm_viterbi import viterbi_decode_trip

app = FastAPI(title="PathSnap API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dataset paths
STUDENT_TEST_DIR = r"C:\Users\admin\Downloads\Student folder\Student folder\student_test"
EVALUATOR_TEST_DIR = r"C:\Users\admin\.gemini\antigravity\scratch\PathSnap\evaluator_test\evaluator_test"

# Use evaluator_test as primary evaluation partition if available
if os.path.exists(EVALUATOR_TEST_DIR) and len(os.listdir(EVALUATOR_TEST_DIR)) > 0:
    TEST_DIR = EVALUATOR_TEST_DIR
    PARTITION_NAME = "evaluator_test"
else:
    TEST_DIR = STUDENT_TEST_DIR
    PARTITION_NAME = "student_test"

MODELS_DIR = r"C:\Users\admin\.gemini\antigravity\scratch\PathSnap\models"
MODEL_PATH = os.path.join(MODELS_DIR, "best_tuned_histgradientboosting.joblib")

# Load model globally into memory
print("Loading PathSnap tuned model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# Available test trips
AVAILABLE_TRIPS = sorted(os.listdir(TEST_DIR))
print(f"Loaded {len(AVAILABLE_TRIPS)} test trips from {PARTITION_NAME}: {AVAILABLE_TRIPS}")

class EvaluationRequest(BaseModel):
    trip_id: str
    use_hmm: bool = True

class RouteSegment(BaseModel):
    arc_id: int
    u_lon: float
    u_lat: float
    v_lon: float
    v_lat: float

class EvaluationResponse(BaseModel):
    trip_id: str
    n_observations: int
    top1_accuracy: float
    route_continuity: float
    sequence_jaccard: float
    inference_time_ms: float
    predicted_arcs_sample: List[int]
    ground_truth_arcs_sample: List[int]
    predicted_segments: List[RouteSegment]
    ground_truth_segments: List[RouteSegment]
    gps_points_sample: List[List[float]]

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "PathSnap ML Evaluation API",
        "model": "HistGradientBoosting + Bayesian TPE + Viterbi HMM",
        "available_trips": AVAILABLE_TRIPS
    }

@app.get("/trips")
def get_trips():
    """Returns list of 10 evaluation trip IDs available for testing."""
    return {"trips": AVAILABLE_TRIPS}

@app.post("/predict_and_evaluate", response_model=EvaluationResponse)
def predict_and_evaluate(req: EvaluationRequest):
    trip_id = req.trip_id
    if trip_id not in AVAILABLE_TRIPS:
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found in evaluation dataset.")
        
    tdir = os.path.join(TEST_DIR, trip_id)
    df_tr, df_nd, df_ar, df_rt = load_trip_data(tdir, trip_id)
    
    t0 = time.time()
    clf_arcs, hmm_arcs = viterbi_decode_trip(df_tr, df_nd, df_ar, model)
    inf_time_ms = (time.time() - t0) * 1000.0
    
    final_pred = np.array(hmm_arcs if req.use_hmm else clf_arcs)
    
    # Server-side Ground Truth calculation (never leaked raw to client before request)
    scale = get_projection_scale(df_tr['lat'].mean())
    gt_arcs = derive_ground_truth_labels(df_tr, df_nd, df_ar, df_rt, scale)
    
    # Calculate evaluation metrics
    acc = float(np.mean(final_pred == gt_arcs))
    
    # Route continuity
    u_idx = df_ar['u'].values
    v_idx = df_ar['v'].values
    valid_trans = 0
    total_trans = len(final_pred) - 1
    for i in range(total_trans):
        a1, a2 = final_pred[i], final_pred[i+1]
        if a1 == a2 or v_idx[a1] == u_idx[a2]:
            valid_trans += 1
    continuity = float(valid_trans / max(1, total_trans))
    
    # Sequence Jaccard
    gt_unique = set(gt_arcs)
    pred_unique = set(final_pred[final_pred >= 0])
    jaccard = float(len(pred_unique & gt_unique) / max(1, len(pred_unique | gt_unique)))
    
    # Format segment coordinates for UI rendering
    nodes = df_nd[['lon', 'lat']].values
    
    # Build unique predicted segments
    pred_segments = []
    seen_pred = set()
    for a in final_pred:
        if a >= 0 and a not in seen_pred:
            seen_pred.add(a)
            u, v = u_idx[a], v_idx[a]
            pred_segments.append(RouteSegment(
                arc_id=int(a),
                u_lon=float(nodes[u, 0]),
                u_lat=float(nodes[u, 1]),
                v_lon=float(nodes[v, 0]),
                v_lat=float(nodes[v, 1])
            ))
            
    gt_segments = []
    seen_gt = set()
    for a in df_rt['arc_id'].values:
        if a not in seen_gt:
            seen_gt.add(a)
            u, v = u_idx[a], v_idx[a]
            gt_segments.append(RouteSegment(
                arc_id=int(a),
                u_lon=float(nodes[u, 0]),
                u_lat=float(nodes[u, 1]),
                v_lon=float(nodes[v, 0]),
                v_lat=float(nodes[v, 1])
            ))
            
    # Sample GPS observations for display (first 250 points or subsampled)
    sub_step = max(1, len(df_tr) // 300)
    gps_sample = df_tr[['lat', 'lon']].iloc[::sub_step].values.tolist()
    
    print(f"[API INFERENCE LOG] Trip: {trip_id} | Points: {len(df_tr)} | Acc: {acc*100:.2f}% | Continuity: {continuity*100:.2f}% | Latency: {inf_time_ms:.1f}ms")
    
    return EvaluationResponse(
        trip_id=trip_id,
        n_observations=len(df_tr),
        top1_accuracy=round(acc * 100.0, 2),
        route_continuity=round(continuity * 100.0, 2),
        sequence_jaccard=round(jaccard * 100.0, 2),
        inference_time_ms=round(inf_time_ms, 2),
        predicted_arcs_sample=final_pred[:10].tolist(),
        ground_truth_arcs_sample=gt_arcs[:10].tolist(),
        predicted_segments=pred_segments,
        ground_truth_segments=gt_segments,
        gps_points_sample=gps_sample
    )

if __name__ == "__main__":
    import uvicorn
    # Can run directly on 0.0.0.0:8000 for network access across laptops
    uvicorn.run(app, host="0.0.0.0", port=8000)
