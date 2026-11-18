"""Contract inspected from user-supplied Parquet. No legacy feature approximation."""
import numpy as np
import pandas as pd
from backend.app.core.errors import AppError

SCHEMA_VERSION = 'notebook-features-v2'
PIPELINE_VERSION = 'authoritative-notebook-v2'
FEATURE_TYPES = {
    'dist_to_edge_m': 'float64', 'dist_to_u_node_m': 'float64', 'dist_to_v_node_m': 'float64',
    'bearing_diff_deg': 'float64', 'bearing_valid': 'int64', 'time_gap_prev_s': 'float64',
    'speed_implied_mps': 'float64', 'irregular_gap_flag': 'float64', 'candidate_rank': 'int64',
    'dist_minus_best_at_point_m': 'float64', 'num_candidates_at_point': 'int64',
    'edge_length_m': 'float64', 'u_node_degree': 'int64', 'v_node_degree': 'int64',
    'reverse_edge_exists': 'int64',
}
FEATURES = list(FEATURE_TYPES)
METADATA = ['trip_id', 'point_idx', 'candidate_edge_id', 'timestamp']
NULLABLE = {'bearing_diff_deg', 'time_gap_prev_s', 'speed_implied_mps', 'irregular_gap_flag'}
SCALER_ORDER = FEATURES.copy()
# The notebook's explicit FEATURE_COLUMNS is authoritative for models/scaler.
# Parquet physical storage order is intentionally separate.
PARQUET_FEATURES = FEATURES.copy()
PARQUET_FEATURES[3:5] = ['bearing_valid', 'bearing_diff_deg']

def validate_features(frame, allow_missing=False):
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != FEATURES:
        raise AppError('INVALID_FEATURE_SCHEMA', 'Feature names and order must match the canonical contract')
    if frame.empty:
        raise AppError('EMPTY_CANDIDATES', 'No candidate rows available')
    for col, dtype in FEATURE_TYPES.items():
        if str(frame[col].dtype) != dtype:
            raise AppError('INVALID_FEATURE_SCHEMA', 'Feature data type mismatch', details={'feature': col, 'expected': dtype})
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise AppError('INVALID_FEATURE_VALUES', 'Infinite feature values are forbidden')
    allowed = NULLABLE if allow_missing else set()
    if any(frame[c].isna().any() for c in FEATURES if c not in allowed):
        raise AppError('INVALID_FEATURE_VALUES', 'Unexpected missing feature values')
    return frame

def group_scores(frame, mapping, scores, observation_count, arc_count):
    validate_features(frame, allow_missing=True)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise AppError('INVALID_MODEL_OUTPUT', 'Expected one finite binary probability per candidate row')
    validate_mapping(mapping, len(frame), observation_count, arc_count)
    groups = [[] for _ in range(observation_count)]
    for row, (obs, arc) in enumerate(mapping.itertuples(index=False, name=None)):
        groups[obs].append({'arc_id': int(arc), 'probability': float(scores[row]), 'row_index': row})
    if not groups or any(not g for g in groups):
        raise AppError('EMPTY_CANDIDATE_GROUP', 'Every observation must have candidates')
    # Ranking preserves the original row/arc association; ties retain dataset row order.
    return [sorted(g, key=lambda c: -c['probability']) for g in groups]


def validate_mapping(mapping, row_count, observation_count, arc_count):
    if len(mapping) != row_count or list(mapping.columns) != ['point_idx', 'candidate_edge_id']:
        raise AppError('INVALID_CANDIDATE_MAPPING', 'Candidate mapping must align exactly with score rows')
    if any(str(mapping[c].dtype) != 'int64' for c in mapping):
        raise AppError('INVALID_CANDIDATE_MAPPING', 'Candidate identifiers must be int64')
    if mapping.duplicated().any() or ((mapping.point_idx < 0) | (mapping.point_idx >= observation_count)).any() or ((mapping.candidate_edge_id < 0) | (mapping.candidate_edge_id >= arc_count)).any():
        raise AppError('INVALID_CANDIDATE_MAPPING', 'Duplicate or out-of-range candidate identifiers')
    counts = mapping.groupby('point_idx').size()
    if observation_count <= 0 or len(counts) != observation_count:
        raise AppError('EMPTY_CANDIDATE_GROUP', 'Every observation must have candidates')
