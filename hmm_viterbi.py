"""
HMM and Viterbi Route Reconstruction for PathSnap

State space:
- For each GPS observation t: hidden states are candidate arcs C_t.
- Emission probability P(z_t | c_t) is derived from the calibrated classifier score / probability.
  Specifically: log P(z_t | c_t) = log(p_clf) or Gaussian spatial emission based on dist(z_t, c_t).
- Transition probability P(c_t | c_{t-1}) enforces road-network connectivity and path plausibility:
  If arc c_{t-1} and c_t are adjacent in road network (v(c_{t-1}) == u(c_t)) or identical (c_{t-1} == c_t):
    high transition probability.
  Otherwise, computed based on network shortest path distance d_net vs Euclidean/spherical distance d_gps:
    P(c_t | c_{t-1}) ~ exp(-|d_net - d_gps| / beta).
- Viterbi recurrence finds the most probable sequence of road segments:
  V[t, c_t] = max_{c_{t-1}} { V[t-1, c_{t-1}] + log P(c_t | c_{t-1}) } + log P(z_t | c_t).
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from feature_engineering import load_trip_data, extract_features_for_trip, get_projection_scale

CACHE_DIR = r"C:\Users\admin\.gemini\antigravity\scratch\PathSnap\data_cache"
MODELS_DIR = r"C:\Users\admin\.gemini\antigravity\scratch\PathSnap\models"
TEST_DIR = r"C:\Users\admin\Downloads\Student folder\Student folder\student_test"

def build_road_graph(df_arcs, df_nodes, scale):
    """
    Builds a directed graph of the road network where edge weights are segment lengths.
    """
    nodes = df_nodes[['lon', 'lat']].values
    u_idx = df_arcs['u'].values
    v_idx = df_arcs['v'].values
    u_m = nodes[u_idx] * scale
    v_m = nodes[v_idx] * scale
    lengths = np.sqrt(np.sum((v_m - u_m)**2, axis=1))
    
    G = nx.DiGraph()
    # Also build arc-to-arc adjacency mapping for fast lookups
    arc_to_nodes = {}
    node_out_arcs = {} # u -> list of (arc_id, v, length)
    node_in_arcs = {}  # v -> list of (arc_id, u, length)
    
    for arc_id in range(len(df_arcs)):
        u = int(u_idx[arc_id])
        v = int(v_idx[arc_id])
        w = float(lengths[arc_id])
        arc_to_nodes[arc_id] = (u, v, w)
        if u not in node_out_arcs:
            node_out_arcs[u] = []
        node_out_arcs[u].append((arc_id, v, w))
        
    return arc_to_nodes, node_out_arcs, lengths

def viterbi_decode_trip(df_track, df_nodes, df_arcs, model, max_candidates=15, beta=10.0, sigma=15.0):
    """
    Runs candidate prediction with trained classifier, then decodes optimal route via Viterbi HMM.
    """
    scale = get_projection_scale(df_track['lat'].mean())
    nodes = df_nodes[['lon', 'lat']].values
    tr_m = df_track[['lon', 'lat']].values * scale
    N_pts = len(df_track)
    
    # 1. Extract features and predict candidate probabilities
    X_trip, _, cand_info, _ = extract_features_for_trip(df_track, df_nodes, df_arcs, df_route=None, max_candidates=max_candidates)
    probs = model.predict_proba(X_trip)[:, 1]
    
    # Organize candidates by observation index
    # obs_candidates[t] = list of (arc_id, clf_prob, dist_m)
    obs_candidates = [[] for _ in range(N_pts)]
    for row_idx, (obs_idx, arc_id) in enumerate(cand_info):
        dist_m = X_trip.iloc[row_idx]['dist_to_segment']
        obs_candidates[obs_idx].append({
            'arc_id': arc_id,
            'prob': probs[row_idx],
            'dist': dist_m
        })
        
    # Precompute road arcs lookup
    u_idx = df_arcs['u'].values
    v_idx = df_arcs['v'].values
    u_m = nodes[u_idx] * scale
    v_m = nodes[v_idx] * scale
    arc_lens = np.sqrt(np.sum((v_m - u_m)**2, axis=1))
    
    # Viterbi tables
    # V[t] will be a list of log-probabilities for each candidate at step t
    # Backtrack[t] stores best predecessor candidate index
    V = []
    backtrack = []
    
    # Classifier-only baseline predictions (argmax at each step)
    clf_only_arcs = []
    for t in range(N_pts):
        cands = obs_candidates[t]
        if not cands:
            clf_only_arcs.append(-1)
        else:
            best_c = max(cands, key=lambda c: c['prob'])
            clf_only_arcs.append(best_c['arc_id'])
            
    # Initial step t=0
    cands_0 = obs_candidates[0]
    if not cands_0:
        # Fallback if no candidate found
        return clf_only_arcs, clf_only_arcs
        
    v0 = []
    for c in cands_0:
        # Emission log-prob: combination of classifier probability and spatial distance
        # log P(z|c) = log(prob + 1e-4) - (dist^2) / (2 * sigma^2)
        em_log = np.log(max(1e-4, c['prob'])) - (c['dist']**2) / (2.0 * sigma**2)
        v0.append(em_log)
    V.append(np.array(v0))
    backtrack.append([None] * len(cands_0))
    
    # Sequential forward pass
    for t in range(1, N_pts):
        prev_cands = obs_candidates[t-1]
        curr_cands = obs_candidates[t]
        v_prev = V[t-1]
        
        # GPS distance between consecutive observations
        d_gps = np.linalg.norm(tr_m[t] - tr_m[t-1])
        dt = max(0.1, df_track['time'].iloc[t] - df_track['time'].iloc[t-1])
        
        v_curr = []
        bp_curr = []
        
        for j, c_j in enumerate(curr_cands):
            arc_j = c_j['arc_id']
            u_j, v_j = u_idx[arc_j], v_idx[arc_j]
            
            # Emission score
            em_log = np.log(max(1e-4, c_j['prob'])) - (c_j['dist']**2) / (2.0 * sigma**2)
            
            best_val = -1e9
            best_k = 0
            
            for k, c_k in enumerate(prev_cands):
                arc_k = c_k['arc_id']
                u_k, v_k = u_idx[arc_k], v_idx[arc_k]
                
                # Transition model:
                # Case 1: Same arc
                if arc_k == arc_j:
                    trans_cost = 0.0
                # Case 2: Directly connected in network (v_k == u_j)
                elif v_k == u_j:
                    trans_cost = 0.5
                # Case 3: Share a junction (opposite direction or sibling)
                elif u_k == u_j or v_k == v_j or u_k == v_j:
                    trans_cost = 3.0
                else:
                    # Disconnected jump penalty based on spatial discrepancy
                    gap = np.linalg.norm(u_m[arc_j] - v_m[arc_k])
                    trans_cost = 5.0 + (gap / beta)
                    
                score = v_prev[k] - trans_cost + em_log
                if score > best_val:
                    best_val = score
                    best_k = k
                    
            v_curr.append(best_val)
            bp_curr.append(best_k)
            
        V.append(np.array(v_curr))
        backtrack.append(bp_curr)
        
    # Backtrack optimal sequence
    best_last_idx = int(np.argmax(V[-1]))
    decoded_arcs = [None] * N_pts
    curr_idx = best_last_idx
    
    for t in range(N_pts - 1, -1, -1):
        decoded_arcs[t] = obs_candidates[t][curr_idx]['arc_id']
        curr_idx = backtrack[t][curr_idx]
        
    return clf_only_arcs, decoded_arcs

def evaluate_route_reconstruction():
    print("Loading tuned model for HMM route reconstruction...")
    model = joblib.load(os.path.join(MODELS_DIR, "best_tuned_histgradientboosting.joblib"))
    
    test_trips = sorted(os.listdir(TEST_DIR))
    print(f"Evaluating route reconstruction on all {len(test_trips)} student-test trips...")
    
    results = []
    
    for tid in test_trips:
        tdir = os.path.join(TEST_DIR, tid)
        df_tr, df_nd, df_ar, df_rt = load_trip_data(tdir, tid)
        
        # Ground truth
        scale = get_projection_scale(df_tr['lat'].mean())
        from feature_engineering import derive_ground_truth_labels
        gt_arcs = derive_ground_truth_labels(df_tr, df_nd, df_ar, df_rt, scale)
        
        t0 = time.time()
        clf_arcs, hmm_arcs = viterbi_decode_trip(df_tr, df_nd, df_ar, model)
        elapsed = time.time() - t0
        
        clf_arcs = np.array(clf_arcs)
        hmm_arcs = np.array(hmm_arcs)
        
        # Observation Accuracy
        clf_acc = np.mean(clf_arcs == gt_arcs)
        hmm_acc = np.mean(hmm_arcs == gt_arcs)
        
        # Route continuity (% identical or connected consecutive transitions)
        u_idx = df_ar['u'].values
        v_idx = df_ar['v'].values
        
        def calc_continuity(seq):
            if len(seq) <= 1:
                return 1.0
            valid = 0
            for i in range(len(seq) - 1):
                a1, a2 = seq[i], seq[i+1]
                if a1 == a2 or v_idx[a1] == u_idx[a2]:
                    valid += 1
            return valid / (len(seq) - 1)
            
        clf_cont = calc_continuity(clf_arcs)
        hmm_cont = calc_continuity(hmm_arcs)
        
        # Sequence-level match: Edit distance or unique traversed arcs overlap (Jaccard similarity)
        gt_unique = set(gt_arcs)
        clf_jaccard = len(set(clf_arcs) & gt_unique) / max(1, len(set(clf_arcs) | gt_unique))
        hmm_jaccard = len(set(hmm_arcs) & gt_unique) / max(1, len(set(hmm_arcs) | gt_unique))
        
        print(f"Trip {tid}:")
        print(f"  Classifier-only: Acc={clf_acc*100:.2f}%, Continuity={clf_cont*100:.2f}%, Jaccard={clf_jaccard*100:.2f}%")
        print(f"  HMM/Viterbi:     Acc={hmm_acc*100:.2f}%, Continuity={hmm_cont*100:.2f}%, Jaccard={hmm_jaccard*100:.2f}% (Decoded in {elapsed:.2f}s)")
        
        results.append({
            'trip_id': tid,
            'n_points': len(df_tr),
            'clf_accuracy': clf_acc,
            'hmm_accuracy': hmm_acc,
            'clf_continuity': clf_cont,
            'hmm_continuity': hmm_cont,
            'clf_jaccard': clf_jaccard,
            'hmm_jaccard': hmm_jaccard,
            'decode_time_s': elapsed
        })
        
    df_eval = pd.DataFrame(results)
    eval_csv = os.path.join(CACHE_DIR, "hmm_route_reconstruction_results.csv")
    df_eval.to_csv(eval_csv, index=False)
    
    print("\n--- OVERALL ROUTE RECONSTRUCTION SUMMARY ---")
    summary = pd.DataFrame([
        {
            "Method": "Classifier-Only Predictions",
            "Mean Top-1 Accuracy": f"{df_eval['clf_accuracy'].mean()*100:.2f}%",
            "Mean Route Continuity": f"{df_eval['clf_continuity'].mean()*100:.2f}%",
            "Mean Sequence Jaccard": f"{df_eval['clf_jaccard'].mean()*100:.2f}%"
        },
        {
            "Method": "Classifier + HMM/Viterbi Decoding",
            "Mean Top-1 Accuracy": f"{df_eval['hmm_accuracy'].mean()*100:.2f}%",
            "Mean Route Continuity": f"{df_eval['hmm_continuity'].mean()*100:.2f}%",
            "Mean Sequence Jaccard": f"{df_eval['hmm_jaccard'].mean()*100:.2f}%"
        }
    ])
    print(summary)
    summary.to_csv(os.path.join(CACHE_DIR, "hmm_summary_comparison.csv"), index=False)

if __name__ == "__main__":
    evaluate_route_reconstruction()
