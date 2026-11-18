"""Real trained LightGBM model, end to end. Skipped when the model library or data are absent."""
import json
from time import sleep, monotonic
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.app.config import ROOT, Settings
from backend.app.main import create_app
from backend.app.models.contract import FEATURES
from backend.app.models.lightgbm_provider import LightGBMModelProvider

lightgbm = pytest.importorskip('lightgbm')
FEATURE_DIR = ROOT/'data/authoritative'
MANIFEST = ROOT/'lightgbm/models/model_manifest.json'
needs_data = pytest.mark.skipif(not (FEATURE_DIR/'student_test_features.parquet').is_file(),
                                reason='Run scripts/import_dataset.py to install data/authoritative')


@needs_data
def test_real_model_reproduces_training_notebook_probabilities():
    provider = LightGBMModelProvider(MANIFEST, FEATURE_DIR); provider.load()
    assert provider.is_ready(), provider.reason
    features = pd.read_parquet(FEATURE_DIR/'student_test_features.parquet')
    features['trip_key'] = features.trip_id.map(lambda t: str(t).zfill(8))
    saved = pd.read_parquet(ROOT/'lightgbm/results/lightgbm_student_test_proba.parquet')
    scored = features[['trip_key', 'point_idx', 'candidate_edge_id']].assign(p=provider.predict_candidate_scores(features[FEATURES]))
    merged = scored.merge(saved, on=['trip_key', 'point_idx', 'candidate_edge_id'], validate='one_to_one')
    assert len(merged) == len(features)
    assert np.abs(merged.p - merged.proba).max() < 1e-9


def test_supplied_feature_scaler_is_not_the_model_preprocessing():
    # Guard against regressing to feature_scaler.joblib, whose ~1e-12 drift flips tree splits.
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['preprocessing_path'] == 'lightgbm_preprocessing.json'
    assert manifest['feature_name_binding'] == 'positional'


@needs_data
def test_real_reconstruction_reports_server_side_accuracy():
    with TestClient(create_app(Settings(manifest_path=MANIFEST, feature_dir=FEATURE_DIR))) as client:
        assert client.get('/api/v1/health/ready').json()['model_ready'] is True
        start = client.post('/api/v1/reconstructions', json={'trip_id': '00000023', 'use_hmm': True})
        assert start.status_code == 202, start.text
        job_id, deadline = start.json()['job_id'], monotonic() + 60
        while client.get(f'/api/v1/reconstructions/{job_id}').json()['status'] not in ['COMPLETED', 'FAILED'] and monotonic() < deadline:
            sleep(.05)
        result = client.get(f'/api/v1/reconstructions/{job_id}/result')
        assert result.status_code == 200, result.text
        metrics = result.json()['metrics']
        for key in ['top1_accuracy', 'classifier_top1_accuracy', 'candidate_miss_rate', 'route_arc_precision', 'route_arc_recall']:
            assert 0 <= metrics[key] <= 1
        assert metrics['evaluated_points'] > 0
        # Reference arcs never leave the server.
        assert all(word not in result.text.lower() for word in ['aligned_edge', 'alignment_status', 'reference_route', '"label"'])
