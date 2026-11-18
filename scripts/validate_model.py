"""Validate a supplied trained model; never trains or substitutes weights."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.app.config import Settings
from backend.app.models.lightgbm_provider import LightGBMModelProvider
from backend.app.services.feature_service import FeatureService
from backend.app.services.trip_service import TripService

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest',type=Path)
parser.add_argument('--trip',help='Also validate exact features bound to this trip')
args=parser.parse_args()
settings=Settings()
provider=LightGBMModelProvider(args.manifest or settings.manifest_path,settings.feature_dir)
provider.load()
print(json.dumps(provider.get_metadata(),indent=2))
if not provider.is_ready(): raise SystemExit(1)
if args.trip:
    trip=TripService(settings).load(args.trip)
    x,mapping=FeatureService(settings.feature_dir).prepare(trip)
    scores=provider.predict_candidate_scores(x)
    print(json.dumps({'trip_id':args.trip,'candidate_rows':len(x),'score_count':len(scores)}))
