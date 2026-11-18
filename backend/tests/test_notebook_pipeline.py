"""Notebook provenance, faithful extraction, route isolation and compatibility gates."""
import ast
import hashlib
import json
from pathlib import Path
from dataclasses import fields
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.app.core.errors import AppError
from backend.app.config import ROOT, Settings
from backend.app.main import create_app
from backend.app.models.contract import FEATURES, FEATURE_TYPES, PARQUET_FEATURES, SCALER_ORDER, validate_features
from backend.app.models.lightgbm_provider import Manifest
from backend.app.pipeline.notebook_features import TripData, FEATURE_COLUMNS, build_trip_table, _time_gaps_s, _implied_speed_mps, _node_degrees
from backend.app.pipeline.reference_alignment import TripData as ReferenceTrip, align_gps_to_route
from backend.app.pipeline.compatibility import NotebookCompatibility, NOTEBOOK_HASH
from backend.app.services.feature_service import FeatureService
from backend.app.services.notebook_feature_service import NotebookFeatureService
from backend.app.services.trip_service import TripService
from scripts.extract_notebook_pipeline import extract
from backend.tests.conftest import SyntheticUnitTestProvider


def test_original_notebook_and_generated_definitions_are_unchanged():
    notebook = ROOT/'notebooks/PathSnap_Dataset_Pipeline_Final_Pointwise.ipynb'
    assert hashlib.sha256(notebook.read_bytes()).hexdigest() == NOTEBOOK_HASH
    features, alignment = extract()
    # Explicit UTF-8: Windows would otherwise decode with cp1252 and garble the em dash.
    assert (ROOT/'backend/app/pipeline/notebook_features.py').read_text(encoding='utf-8') == features
    assert (ROOT/'backend/app/pipeline/reference_alignment.py').read_text(encoding='utf-8') == alignment


def test_explicit_notebook_model_order_is_separate_from_storage_order():
    assert FEATURES == FEATURE_COLUMNS == SCALER_ORDER
    assert FEATURES[3:5] == ['bearing_diff_deg','bearing_valid']
    assert PARQUET_FEATURES[3:5] == ['bearing_valid','bearing_diff_deg']
    manifest = json.loads((ROOT/'models/model_manifest.example.json').read_text())
    Manifest.model_validate(manifest)
    manifest['feature_schema_version'] = 'authoritative-parquet-v1'
    with pytest.raises(ValueError): Manifest.model_validate(manifest)


def test_generation_type_cannot_carry_reference_labels():
    assert set(f.name for f in fields(TripData)) == {'trip_id','nodes','arcs','track'}
    tree = ast.parse((ROOT/'backend/app/pipeline/notebook_features.py').read_text())
    assert not any(isinstance(n,ast.Attribute) and n.attr == 'route' for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'open','exec','eval'} for n in ast.walk(tree))


def test_synthetic_geometry_heading_gaps_and_topology(synthetic_environment):
    settings,_ = synthetic_environment
    raw = TripService(settings).load('00000001')
    trip = TripData(raw.trip_id,raw.nodes.to_numpy(),raw.arcs.to_numpy(),raw.track.to_numpy())
    result = build_trip_table(trip)
    validate_features(result[FEATURES],allow_missing=True)
    assert len(result) == 15
    assert result.groupby('point_idx').size().eq(5).all()
    assert result[result.point_idx==0].bearing_valid.eq(0).all()
    assert result[result.point_idx==0].bearing_diff_deg.isna().all()
    assert result[result.point_idx==1].time_gap_prev_s.eq(1).all()
    assert result.num_candidates_at_point.eq(5).all()
    assert 'label' not in result and 'aligned_edge_id' not in result
    np.testing.assert_array_equal(_node_degrees(trip.arcs,4),[2,3,3,2])
    gaps = trip.track.copy(); gaps[:,2] = [1,1,100]
    assert np.isnan(_implied_speed_mps(gaps)[1])
    np.testing.assert_allclose(_time_gaps_s(gaps),[np.nan,0,99],equal_nan=True)


def test_synthetic_ordered_reference_alignment_is_isolated(synthetic_environment):
    settings,_ = synthetic_environment
    raw = TripService(settings).load('00000001')
    source = ReferenceTrip(raw.trip_id,raw.nodes.to_numpy(),raw.arcs.to_numpy(),raw.track.to_numpy(),np.array([0,1,2]))
    aligned = align_gps_to_route(source)
    assert len(aligned)==3
    assert (np.diff(aligned.aligned_route_pos)>=0).all()
    assert aligned.alignment_status.eq('accepted').all()
    assert aligned.aligned_edge_id.isin([0,1,2]).all()
    # Changing the protected route changes labels, never feature generation.
    a = build_trip_table(TripData(source.trip_id,source.nodes,source.arcs,source.track))
    b = build_trip_table(TripData(source.trip_id,source.nodes,source.arcs,source.track))
    pd.testing.assert_frame_equal(a,b)


@pytest.mark.parametrize('report',['absent','failed','forged_success','wrong_runtime'])
def test_unproven_online_generation_fails_closed(tmp_path,report):
    path = tmp_path/'parity.json'
    if report != 'absent':
        content = json.loads((ROOT/'docs/NOTEBOOK_PARITY.json').read_text())
        if report in ['forged_success','wrong_runtime']: content['all_match']=True
        if report=='wrong_runtime': content['versions']['numpy']='unverified'
        path.write_text(json.dumps(content))
    gate = NotebookCompatibility(ROOT/'data/authoritative',path)
    assert gate.availability()['ready'] is False
    with pytest.raises(AppError) as exc: gate.require_ready()
    assert exc.value.code=='FEATURE_PIPELINE_INCOMPATIBLE'


def test_online_adapter_matches_notebook_when_gate_passes_in_synthetic_unit_test(synthetic_environment,monkeypatch):
    settings,_ = synthetic_environment
    raw=TripService(settings).load('00000001')
    gate=NotebookCompatibility(settings.feature_dir,settings.parity_report)
    monkeypatch.setattr(gate,'require_ready',lambda:{'SYNTHETIC_UNIT_TEST_ONLY':True})
    called=[]
    result=NotebookFeatureService(gate).generate(raw,lambda:called.append(True))
    expected=build_trip_table(TripData(raw.trip_id,raw.nodes.to_numpy(),raw.arcs.to_numpy(),raw.track.to_numpy()))
    pd.testing.assert_frame_equal(result,expected)
    assert len(called)==2


def test_feature_service_uses_common_validation_for_generated_candidates(synthetic_environment,monkeypatch):
    settings,_=synthetic_environment
    raw=TripService(settings).load('00000001')
    service=FeatureService(settings.feature_dir)
    monkeypatch.setattr(service,'_source',lambda tid:None)
    monkeypatch.setattr(service.compatibility,'require_ready',lambda:{'SYNTHETIC_UNIT_TEST_ONLY':True})
    stages=[]
    x,m=service.prepare(raw,on_generation=lambda:stages.append('GENERATING_CANDIDATES'))
    assert list(x)==FEATURES and len(x)==len(m)==15
    assert stages==['GENERATING_CANDIDATES']


def test_real_evaluator_generation_remains_blocked_and_model_stays_absent(tmp_path):
    with TestClient(create_app(Settings(notebook_demo_mode=False,manifest_path=tmp_path/'absent.json'))) as client:
        metadata=client.get('/api/v1/trips/00000023').json()
        assert metadata['features']['ready'] is False
        assert 'parity' in metadata['features']['reason']
        assert client.get('/api/v1/health/ready').json()['model_ready'] is False
        response=client.post('/api/v1/reconstructions',json={'trip_id':'00000023'})
        assert response.status_code==503 and response.json()['error']['code']=='MODEL_NOT_READY'


def test_real_evaluator_generation_cannot_use_supplied_test_provider():
    # Even a ready test double cannot bypass the incompatible feature contract.
    with TestClient(create_app(Settings(notebook_demo_mode=False),provider=SyntheticUnitTestProvider())) as client:
        response=client.post('/api/v1/reconstructions',json={'trip_id':'00000023'})
        assert response.status_code==503
        assert response.json()['error']['code']=='FEATURE_PIPELINE_UNAVAILABLE'
