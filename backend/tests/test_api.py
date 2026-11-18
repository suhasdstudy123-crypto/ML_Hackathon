from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import sleep,monotonic
import json
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.app.main import create_app
from backend.app.config import Settings
from backend.app.core.errors import AppError
from backend.app.services.trip_service import TripService,arc_geometry
from backend.app.services.feature_service import FeatureService
from backend.tests.conftest import SyntheticUnitTestProvider

FORBIDDEN=['ground_truth','aligned_edge','aligned_route','alignment_status','usable_for_training','"label"','reference_route']

def wait_done(client,jid):
    deadline=monotonic()+5
    while monotonic()<deadline:
        j=client.get('/api/v1/reconstructions/'+jid).json()
        if j['status'] in ['COMPLETED','FAILED']:return j
        sleep(.01)
    raise AssertionError('Job did not finish')

def test_real_ten_trip_preview_and_no_reference_exposure(tmp_path):
    # Without a configured model the previews still work and reconstruction is refused.
    app=create_app(Settings(manifest_path=tmp_path/'absent.json'))
    with TestClient(app) as c:
        assert c.get('/api/v1/health/live').status_code==200
        ready=c.get('/api/v1/health/ready').json();assert not ready['model_ready']
        ids=c.get('/api/v1/trips').json()['trips'];assert len(ids)==10
        for tid in ids:
            meta=c.get('/api/v1/trips/'+tid);assert meta.status_code==200
            r=c.get('/api/v1/trips/'+tid+'/preview');assert r.status_code==200
            data=r.json();assert len(data['geometry']['raw_gps']['features'])==meta.json()['observation_count']
            assert data['geometry']['road_network']['features']
            assert all(word not in r.text.lower() for word in FORBIDDEN)
            response=c.post('/api/v1/reconstructions',json={'trip_id':tid})
            assert response.status_code==503 and response.json()['error']['code']=='MODEL_NOT_READY'
        assert c.get('/trips').json()['trips']==ids
        assert c.post('/predict_and_evaluate',json={'trip_id':ids[0]}).status_code==503
        for url in ['/maps/trip_00000000_reconstructed.html','/data_cache/train_meta.csv','/evaluator_test/evaluator_test/00000015/00000015.route']:
            assert c.get(url).status_code==404

@pytest.mark.parametrize('body,status',[({'trip_id':'../00000015'},422),({'trip_id':'99999999'},404),({'trip_id':'00000015','label':1},422),({'trip_id':'00000015','use_hmm':'yes'},422),({},422)])
def test_invalid_requests(body,status):
    with TestClient(create_app()) as c:
        r=c.post('/api/v1/reconstructions',json=body);assert r.status_code==status
        assert r.json()['error']['request_id']==r.headers['X-Request-ID']
        assert 'traceback' not in r.text.lower()

def test_successful_synthetic_inference_evaluates_after_scoring(synthetic_environment,monkeypatch):
    settings,_=synthetic_environment;provider=SyntheticUnitTestProvider();scored=[]
    predict=provider.predict_candidate_scores
    def score(x):scored.append(True);return predict(x)
    monkeypatch.setattr(provider,'predict_candidate_scores',score)
    original=TripService.path
    def guarded(self,tid,ext):
        if ext=='route':assert scored,'Reference read before scoring'
        return original(self,tid,ext)
    monkeypatch.setattr(TripService,'path',guarded)
    with TestClient(create_app(settings,provider)) as c:
        assert c.get('/api/v1/trips/00000001/preview').status_code==200
        start=c.post('/api/v1/reconstructions',json={'trip_id':'00000001'});assert start.status_code==202
        jid=start.json()['job_id'];job=wait_done(c,jid);assert job['status']=='COMPLETED',job
        assert [e['stage'] for e in job['stages']]==['QUEUED','PREPARING_DATA','PREPARING_FEATURES','SCORING_CANDIDATES','DECODING_ROUTE','EVALUATING','COMPLETED']
        r=c.get(f'/api/v1/reconstructions/{jid}/result');assert r.status_code==200
        d=r.json();assert d['predictions']['hmm_arc_sequence']==[0,1,2]
        assert d['metrics']['sequence_jaccard']==1 and d['metrics']['route_arc_precision']==1 and d['metrics']['route_arc_recall']==1
        # Top-1 is computed server-side from the notebook alignment (synthetic points sit on arc ends).
        assert 0<=d['metrics']['top1_accuracy']<=1 and d['metrics']['classifier_top1_accuracy']==d['metrics']['top1_accuracy']
        assert len(d['geometry']['hmm_route']['features'])==3
        assert all(word not in r.text.lower() for word in FORBIDDEN)
        assert d['model']['version']=='SYNTHETIC_UNIT_TEST_ONLY'

def test_bounded_jobs_deduplicate_and_keep_health_responsive(synthetic_environment):
    settings,_=synthetic_environment;settings=replace(settings,max_jobs=1)
    gate=Event()
    class SlowSyntheticProvider(SyntheticUnitTestProvider):
        def predict_candidate_scores(self,x):gate.wait(3);return super().predict_candidate_scores(x)
    with TestClient(create_app(settings,SlowSyntheticProvider())) as c:
        first=c.post('/api/v1/reconstructions',json={'trip_id':'00000001'}).json()
        try:
            with ThreadPoolExecutor(4) as pool:
                replies=list(pool.map(lambda _:c.post('/api/v1/reconstructions',json={'trip_id':'00000001'}).json(),range(4)))
            assert all(r['job_id']==first['job_id'] for r in replies)
            assert c.get('/api/v1/health/live').status_code==200
            assert c.post('/api/v1/reconstructions',json={'trip_id':'00000001','use_hmm':False}).status_code==429
            assert c.get(f"/api/v1/reconstructions/{first['job_id']}/result").status_code==409
        finally:gate.set()
        assert wait_done(c,first['job_id'])['status']=='COMPLETED'

def test_timeout_is_failed_without_a_result(synthetic_environment):
    settings,_=synthetic_environment
    class SlowSyntheticProvider(SyntheticUnitTestProvider):
        def predict_candidate_scores(self,x):sleep(.03);return super().predict_candidate_scores(x)
    with TestClient(create_app(replace(settings,timeout_seconds=.01),SlowSyntheticProvider())) as c:
        jid=c.post('/api/v1/reconstructions',json={'trip_id':'00000001'}).json()['job_id']
        job=wait_done(c,jid);assert job['status']=='FAILED'
        assert job['error']['code']=='INFERENCE_TIMEOUT'
        assert c.get(f'/api/v1/reconstructions/{jid}/result').status_code==409

def test_cors_is_explicit():
    with TestClient(create_app()) as c:
        for origin,status in [('http://localhost:5173',200),('https://untrusted.example',400)]:
            r=c.options('/api/v1/trips',headers={'Origin':origin,'Access-Control-Request-Method':'GET'})
            assert r.status_code==status
        assert c.get('/api/v1/trips',headers={'Origin':'https://untrusted.example'}).headers.get('access-control-allow-origin') is None

def test_empty_data_and_restart_status(tmp_path):
    with TestClient(create_app(Settings(trip_dir=tmp_path))) as c:
        assert c.get('/api/v1/health/ready').status_code==503
        assert c.get('/api/v1/reconstructions/expired').json()['error']['code']=='JOB_NOT_FOUND'

@pytest.mark.parametrize('ext,body,code',[('track','500 50 1\n','INVALID_COORDINATES'),('arcs','0 100\n','INVALID_ARC_INDICES'),('track','nan 50 1\n','CORRUPT_TRIP_DATA')])
def test_corrupt_raw_data(synthetic_environment,ext,body,code):
    settings,_=synthetic_environment
    (settings.trip_dir/'00000001'/f'00000001.{ext}').write_text(body)
    with pytest.raises(AppError) as e:TripService(settings).load('00000001')
    assert e.value.code==code

def test_geometry_preserves_repeated_edges_without_bridges(synthetic_environment):
    settings,_=synthetic_environment;trip=TripService(settings).load('00000001')
    features=arc_geometry(trip,[0,4,0])['features']
    assert [f['properties']['arc_id'] for f in features]==[0,4,0]
    assert features[0]['geometry']['coordinates']==[[8.,50.],[8.001,50.]]
    assert len(features)==3

def test_feature_binding_and_schema_errors(synthetic_environment):
    settings,frame=synthetic_environment;trip=TripService(settings).load('00000001');service=FeatureService(settings.feature_dir)
    x,m=service.prepare(trip);assert list(m)==['point_idx','candidate_edge_id'] and 'label' not in x
    frame['timestamp']=999.;frame.to_parquet(settings.feature_dir/'inference_features.parquet')
    with pytest.raises(AppError,match='timestamps'):service.prepare(trip)
    (settings.feature_dir/'trip_bindings.json').write_text('{}')
    with pytest.raises(AppError,match='bound'):service.prepare(trip)

def test_preserved_artifact_checksums():
    import hashlib
    root=Path(__file__).resolve().parents[2]
    checks=json.loads((root/'docs/ARTIFACT_CHECKSUMS.json').read_text())
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==value for p,value in checks.items())

def test_job_retention_and_shutdown(synthetic_environment):
    from backend.app.core.jobs import JobManager
    from backend.app.services.inference_service import InferenceService
    from backend.app.services.evaluation_service import EvaluationService
    settings,_=synthetic_environment;trips=TripService(settings)
    jobs=JobManager(settings,InferenceService(trips,FeatureService(settings.feature_dir),SyntheticUnitTestProvider()),EvaluationService(trips))
    try:
        j=jobs.submit('00000001',False,'test-request');deadline=monotonic()+5
        while jobs.get(j['job_id'])['status'] not in ['COMPLETED','FAILED'] and monotonic()<deadline:sleep(.01)
        assert jobs.get(j['job_id'])['status']=='COMPLETED'
        with jobs.lock:jobs.jobs[j['job_id']]['_updated']=monotonic()-settings.retention_seconds-1
        with pytest.raises(AppError,match='expired'):jobs.get(j['job_id'])
    finally:jobs.shutdown()
    with pytest.raises(AppError,match='shutting down'):jobs.submit('00000001',False,'test-request')

def test_corrupt_feature_source_fails_with_explicit_error(synthetic_environment):
    settings,_=synthetic_environment
    (settings.feature_dir/'inference_features.parquet').write_bytes(b'SYNTHETIC_CORRUPT_UNIT_TEST')
    with TestClient(create_app(settings,SyntheticUnitTestProvider())) as c:
        response=c.post('/api/v1/reconstructions',json={'trip_id':'00000001'})
        assert response.status_code==503
        assert response.json()['error']['code']=='INVALID_FEATURE_SCHEMA'
        assert 'Traceback' not in response.text
