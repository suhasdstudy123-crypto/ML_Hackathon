"""Loader/API contract tests use a SYNTHETIC UNIT TEST Booster, never trained weights."""
import hashlib
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from backend.app.core.errors import AppError
from backend.app.models.lightgbm_provider import LightGBMModelProvider
from backend.app.models.contract import FEATURES,FEATURE_TYPES,SCALER_ORDER

class SyntheticTestBooster:
    names=FEATURES
    output=None
    objective='binary sigmoid:1'
    def __init__(self,model_file):
        assert open(model_file).read()=='SYNTHETIC_UNIT_TEST_ONLY'
    def num_feature(self):return 15
    def feature_name(self):return self.names
    def dump_model(self):return {'objective':self.objective}
    def predict(self,X,**kwargs):
        assert list(X)==FEATURES
        return np.full(len(X),.5) if self.output is None else self.output

@pytest.fixture
def synthetic_manifest(tmp_path,synthetic_features,monkeypatch):
    synthetic_features.to_parquet(tmp_path/'train_features.parquet')
    artifact=tmp_path/'synthetic.model';artifact.write_text('SYNTHETIC_UNIT_TEST_ONLY')
    m={'configured':True,'name':'lightgbm','version':'SYNTHETIC_UNIT_TEST_ONLY','artifact_format':'lightgbm_native',
       'artifact_path':artifact.name,'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
       'feature_schema_version':'authoritative-parquet-v1','runtime_pipeline':'precomputed-authoritative-v1',
       'feature_names':FEATURES,'feature_dtypes':FEATURE_TYPES,'training_configuration_reference':'SYNTHETIC_UNIT_TEST_ONLY'}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m))
    monkeypatch.setitem(sys.modules,'lightgbm',SimpleNamespace(Booster=SyntheticTestBooster,LGBMClassifier=type('SyntheticTestClassifier',(),{})))
    return path,m

def test_absent_model_is_not_ready(tmp_path,synthetic_features):
    p=LightGBMModelProvider(tmp_path/'missing.json',tmp_path);p.load()
    assert not p.is_ready() and p.get_metadata()['state']=='UNCONFIGURED'
    with pytest.raises(AppError,match='not connected'):p.predict_candidate_scores(synthetic_features)

def test_native_loader_contract(synthetic_manifest,synthetic_features):
    path,_=synthetic_manifest;p=LightGBMModelProvider(path,path.parent);p.load()
    assert p.is_ready()
    assert p.predict_candidate_scores(synthetic_features).shape==(15,)

@pytest.mark.parametrize('change',['missing_file','invalid_file','order','format','count','preprocessing'])
def test_invalid_artifacts_fail_closed(synthetic_manifest,monkeypatch,change):
    path,m=synthetic_manifest
    if change=='missing_file':m['artifact_path']='absent'
    if change=='invalid_file':(path.parent/m['artifact_path']).write_text('corrupt')
    if change=='order':m['feature_names']=list(reversed(FEATURES))
    if change=='format':m['artifact_format']='histgradientboosting'
    if change=='count':monkeypatch.setattr(SyntheticTestBooster,'names',FEATURES[:-1])
    if change=='preprocessing':m.update(missing_value_policy='preprocessor',preprocessing_path='absent')
    path.write_text(json.dumps(m));p=LightGBMModelProvider(path,path.parent);p.load()
    assert not p.is_ready() and p.get_metadata()['state']=='INVALID'

@pytest.mark.parametrize('values',[np.array([np.nan]*5),np.ones((5,2)),np.full(5,1.1)])
def test_probe_rejects_bad_outputs(synthetic_manifest,monkeypatch,values):
    path,_=synthetic_manifest;monkeypatch.setattr(SyntheticTestBooster,'output',values)
    p=LightGBMModelProvider(path,path.parent);p.load();assert not p.is_ready()

def test_preprocessor_explicit_order_is_restored(synthetic_manifest,synthetic_features,monkeypatch):
    path,m=synthetic_manifest
    m.update(missing_value_policy='preprocessor',preprocessing_path='pre.joblib',preprocessing_sha256=hashlib.sha256(b'synthetic').hexdigest(),
             preprocessing_feature_names=SCALER_ORDER,preprocessing_training_reference='SYNTHETIC_UNIT_TEST_ONLY')
    (path.parent/'pre.joblib').write_bytes(b'synthetic');path.write_text(json.dumps(m))
    class SyntheticPreprocessor:
        n_features_in_=15
        def transform(self,X):
            assert list(X)==SCALER_ORDER
            return X.to_numpy()+1
    monkeypatch.setattr('backend.app.models.lightgbm_provider.joblib.load',lambda p:SyntheticPreprocessor())
    p=LightGBMModelProvider(path,path.parent);p.load();assert p.is_ready()
    p.predict_candidate_scores(synthetic_features)
