"""All generated data in this directory is explicitly SYNTHETIC UNIT TEST data."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from backend.app.config import Settings
from backend.app.models.contract import FEATURES, FEATURE_TYPES

@pytest.fixture
def synthetic_features():
    rows = 15
    data = {k:np.zeros(rows,dtype=t) for k,t in FEATURE_TYPES.items()}
    data['candidate_rank']=np.tile(np.arange(5,dtype='int64'),3)
    data['num_candidates_at_point']=np.full(rows,5,dtype='int64')
    return pd.DataFrame(data)[FEATURES]

@pytest.fixture
def synthetic_environment(tmp_path,synthetic_features):
    raw=tmp_path/'trips'/'00000001';raw.mkdir(parents=True)
    files={'track':'8.0 50.0 1\n8.001 50.0 2\n8.002 50.0 3\n',
           'nodes':'8.0 50.0\n8.001 50.0\n8.002 50.0\n8.003 50.0\n',
           'arcs':'0 1\n1 2\n2 3\n1 0\n3 2\n','route':'0\n1\n2\n'}
    for ext,body in files.items(): (raw/f'00000001.{ext}').write_text(body)
    fd=tmp_path/'features';fd.mkdir()
    frame=synthetic_features.copy()
    frame.insert(0,'timestamp',np.repeat([1.,2.,3.],5))
    frame.insert(0,'candidate_edge_id',np.tile(np.arange(5,dtype='int64'),3))
    frame.insert(0,'point_idx',np.repeat(np.arange(3,dtype='int64'),5))
    frame.insert(0,'trip_id','00000001')
    frame['label']=999.  # Forbidden sentinel; must never enter scorer/API.
    frame.to_parquet(fd/'inference_features.parquet')
    (fd/'trip_bindings.json').write_text(json.dumps({'00000001':{ext:hashlib.sha256((raw/f'00000001.{ext}').read_bytes()).hexdigest() for ext in ['track','nodes','arcs']}}))
    return Settings(trip_dir=tmp_path/'trips',feature_dir=fd,manifest_path=tmp_path/'absent.json'),frame

class SyntheticUnitTestProvider:
    def load(self): pass
    def is_ready(self): return True
    def get_metadata(self):
        return {'name':'lightgbm','version':'SYNTHETIC_UNIT_TEST_ONLY','ready':True,'state':'READY','reason':None,
                'feature_schema_version':'authoritative-parquet-v1','runtime_pipeline':'precomputed-authoritative-v1'}
    def predict_candidate_scores(self,features):
        assert list(features)==FEATURES
        return np.array([.9,.1,.1,.1,.1, .1,.9,.1,.1,.1, .1,.1,.9,.1,.1])
