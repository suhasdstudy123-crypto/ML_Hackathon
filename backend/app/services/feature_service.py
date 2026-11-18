import json
from pathlib import Path
import numpy as np
import pandas as pd
from backend.app.core.errors import AppError
from backend.app.models.contract import FEATURES, METADATA, validate_features, validate_mapping

class FeatureService:
    def __init__(self, feature_dir): self.root = Path(feature_dir)

    def _source(self, trip_id):
        for name in ['train_features.parquet', 'student_test_features.parquet', 'inference_features.parquet']:
            p = self.root / name
            if p.is_file():
                ids = pd.read_parquet(p, columns=['trip_id'], filters=[('trip_id', '==', trip_id)])
                if len(ids): return p
        return None

    def availability(self, trip_id, fingerprints=None):
        if self._source(trip_id) is None:
            return {'ready': False, 'reason': 'Exact authoritative candidate features unavailable for this trip; original generator required'}
        if not (self.root / 'trip_bindings.json').is_file():
            return {'ready': False, 'reason': 'Trip-to-feature raw-file checksum bindings required'}
        try:
            bindings = json.loads((self.root / 'trip_bindings.json').read_text())
            binding = bindings.get(trip_id)
            if not isinstance(binding, dict) or set(binding) != {'track','nodes','arcs'} or (fingerprints is not None and binding != fingerprints):
                raise ValueError('Binding mismatch')
        except Exception:
            return {'ready': False, 'reason': 'Valid raw-file fingerprint bindings required for this trip'}
        return {'ready': True, 'reason': None}

    def prepare(self, trip):
        p = self._source(trip.trip_id)
        if p is None: raise AppError('FEATURE_PIPELINE_UNAVAILABLE', 'Exact authoritative features unavailable; supply the original generator or precomputed trip features', 503)
        try:
            bindings = json.loads((self.root / 'trip_bindings.json').read_text())
            if bindings.get(trip.trip_id) != trip.fingerprints: raise ValueError('Mismatch')
        except Exception as exc:
            raise AppError('FEATURE_BINDING_MISMATCH', 'Precomputed features must be bound to the exact raw trip files', 503) from exc
        try:
            # Labels/alignment diagnostics are never read by the inference service.
            data = pd.read_parquet(p, columns=METADATA + FEATURES, filters=[('trip_id', '==', trip.trip_id)])
        except Exception as exc:
            raise AppError('INVALID_FEATURE_SCHEMA', 'Required inference columns unavailable') from exc
        features = data[FEATURES].reset_index(drop=True)
        mapping = data[['point_idx', 'candidate_edge_id']].reset_index(drop=True)
        validate_features(features, allow_missing=True)
        validate_mapping(mapping, len(features), len(trip.track), len(trip.arcs))
        sizes = mapping.groupby('point_idx').size()
        if not sizes.eq(5).all() or not features.num_candidates_at_point.eq(5).all():
            raise AppError('INVALID_CANDIDATE_GROUP', 'Authoritative contract requires five candidates per observation')
        if not data.groupby('point_idx').candidate_rank.apply(lambda s: sorted(s) == list(range(5))).all():
            raise AppError('INVALID_CANDIDATE_GROUP', 'Candidate ranks must be 0 through 4 in every group')
        expected_time = trip.track.time.to_numpy()[mapping.point_idx.to_numpy()]
        if not np.allclose(data.timestamp, expected_time, rtol=0, atol=1e-8):
            raise AppError('FEATURE_BINDING_MISMATCH', 'Feature timestamps do not match raw observation order')
        return features, mapping
