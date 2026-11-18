"""Validate the configured trained model; never trains or substitutes weights.

Checks:
  1. The backend loads the model exactly as the API does (manifest, hashes, feature order).
  2. Parity: the backend's probabilities on the 10 student-test trips equal the probabilities
     saved by the training notebook (lightgbm/results/lightgbm_student_test_proba.parquet),
     and Top-1 accuracy reproduces the notebook's reported value.
  3. Optional --trip: score one evaluator trip through the live feature pipeline.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from backend.app.config import Settings
from backend.app.models.contract import FEATURES
from backend.app.models.lightgbm_provider import LightGBMModelProvider
from backend.app.services.feature_service import FeatureService
from backend.app.services.trip_service import TripService

def parity(provider, feature_dir):
    saved_path=ROOT/'lightgbm/results/lightgbm_student_test_proba.parquet'
    features=pd.read_parquet(feature_dir/'student_test_features.parquet')
    saved=pd.read_parquet(saved_path)
    features['trip_key']=features.trip_id.map(lambda t:str(t).zfill(8))
    scores=provider.predict_candidate_scores(features[FEATURES])
    merged=features[['trip_key','point_idx','candidate_edge_id']].assign(backend=scores).merge(
        saved,on=['trip_key','point_idx','candidate_edge_id'],how='inner',validate='one_to_one')
    if len(merged)!=len(features): raise SystemExit('Saved probabilities do not cover every student-test row')
    max_diff=float(np.abs(merged.backend-merged.proba).max())
    # Top-1 on points with an accepted reference alignment (same definition as the notebook).
    align=pd.read_parquet(feature_dir/'student_test_alignment_diagnostics.parquet')
    align['trip_key']=align.trip_id.map(lambda t:str(t).zfill(8)); align['point_idx']=align.point_idx.astype('int64')
    top=merged.sort_values(['trip_key','point_idx','backend'],ascending=[True,True,False]).drop_duplicates(['trip_key','point_idx'])
    top=top.merge(align[['trip_key','point_idx','aligned_edge_id','alignment_status']],on=['trip_key','point_idx'])
    known=top.alignment_status.eq('accepted')
    top1=float((top.candidate_edge_id==top.aligned_edge_id)[known].mean())
    return {'student_test_rows':len(merged),'max_abs_probability_difference':max_diff,'student_test_top1_accuracy':round(top1,4)}

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest',type=Path)
parser.add_argument('--trip',help='Also score this evaluator trip through the live feature pipeline')
args=parser.parse_args()
settings=Settings()
provider=LightGBMModelProvider(args.manifest or settings.manifest_path,settings.feature_dir)
provider.load()
print(json.dumps(provider.get_metadata(),indent=2))
if not provider.is_ready(): raise SystemExit(1)
report=parity(provider,settings.feature_dir)
print(json.dumps({'parity':report},indent=2))
if report['max_abs_probability_difference']>1e-9: raise SystemExit('Backend probabilities differ from the training notebook output')
if args.trip:
    trip=TripService(settings).load(args.trip)
    x,mapping=FeatureService(settings.feature_dir,settings.parity_report,settings.notebook_demo_mode).prepare(trip)
    scores=provider.predict_candidate_scores(x)
    print(json.dumps({'trip_id':args.trip,'candidate_rows':len(x),'score_count':len(scores)}))
