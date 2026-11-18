"""Compare reviewed notebook algorithms with immutable supplied data; never fit/write datasets."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import scipy
from backend.app.pipeline.notebook_features import TripData, FEATURE_COLUMNS, build_trip_table
from backend.app.pipeline.reference_alignment import TripData as ReferenceTrip, align_gps_to_route

ROOT = Path(__file__).resolve().parents[1]


def compare(expected, actual, columns):
    if len(expected) != len(actual):
        return {'row_count': {'expected': len(expected), 'generated': len(actual)}}
    differences = {}
    for column in columns:
        a, b = expected[column].to_numpy(), actual[column].to_numpy()
        if a.dtype.kind in 'fiu':
            matches = np.isclose(a, b, rtol=1e-12, atol=1e-8, equal_nan=True)
        else:
            matches = a == b
        if not matches.all():
            differences[column] = {'mismatch_count': int((~matches).sum())}
        if expected[column].dtype != actual[column].dtype:
            differences.setdefault(column, {})['dtype'] = [str(expected[column].dtype), str(actual[column].dtype)]
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--feature-dir', type=Path, default=ROOT/'data/authoritative')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/NOTEBOOK_PARITY.json')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    started = time.perf_counter()
    report = {'notebook_sha256': hashlib.sha256((ROOT/'notebooks/PathSnap_Dataset_Pipeline_Final_Pointwise.ipynb').read_bytes()).hexdigest(),
              'feature_order': FEATURE_COLUMNS, 'tolerance': {'rtol': 1e-12, 'atol': 1e-8},
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
              'source_artifact_sha256': {}, 'trips': []}
    for split in ['train', 'student_test']:
        fp = args.feature_dir/f'{split}_features.parquet'
        ap = args.feature_dir/f'{split}_alignment_diagnostics.parquet'
        for p in [fp, ap]: report['source_artifact_sha256'][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        features = pd.read_parquet(fp)
        alignments = pd.read_parquet(ap)
        for tid in sorted(features.trip_id.unique())[:args.limit]:
            tick = time.perf_counter()
            raw = args.raw_root/split/tid
            frames = {ext: np.atleast_2d(np.loadtxt(raw/f'{tid}.{ext}', dtype='int64' if ext == 'arcs' else float)) for ext in ['nodes','arcs','track']}
            route = np.atleast_1d(np.loadtxt(raw/f'{tid}.route', dtype='int64'))
            trip = TripData(trip_id=tid, **frames)
            generated = build_trip_table(trip)
            expected = features[features.trip_id == tid].reset_index(drop=True)
            feature_diff = compare(expected, generated, ['trip_id','point_idx','candidate_edge_id','timestamp']+FEATURE_COLUMNS)
            reference = ReferenceTrip(trip_id=tid, **frames, route=route)
            aligned = align_gps_to_route(reference)
            obs = generated.point_idx.to_numpy()
            matches = generated.candidate_edge_id.to_numpy() == aligned.aligned_edge_id.to_numpy()[obs]
            hits = pd.Series(matches).groupby(generated.point_idx).any().to_numpy()
            aligned['candidate_hit'] = hits
            aligned['usable_for_training'] = aligned.alignment_status.eq('accepted') & hits
            generated['label'] = np.where(aligned.usable_for_training.to_numpy()[obs], matches.astype(float), np.nan)
            expected_alignment = alignments[alignments.trip_id == tid].drop(columns='trip_id').reset_index(drop=True)
            alignment_diff = compare(expected_alignment, aligned, list(aligned))
            label_diff = compare(expected, generated, ['label'])
            passed = not (feature_diff or alignment_diff or label_diff)
            report['trips'].append({'split':split,'trip_id':tid,'observations':len(trip.track),'candidate_rows':len(generated),
                'features_match':not feature_diff,'alignment_match':not alignment_diff,'labels_match':not label_diff,
                'differences':{'features':feature_diff,'alignment':alignment_diff,'labels':label_diff},
                'raw_sha256':{ext:hashlib.sha256((raw/f'{tid}.{ext}').read_bytes()).hexdigest() for ext in ['nodes','arcs','track','route']},
                'duration_seconds':round(time.perf_counter()-tick,4)})
            if len(report['trips']) % 10 == 0 or not passed:
                print(f"Checked {len(report['trips'])} trips; {tid}: {'PASS' if passed else 'MISMATCH'}", flush=True)
    report['trip_count'] = len(report['trips'])
    report['all_match'] = all(t['features_match'] and t['alignment_match'] and t['labels_match'] for t in report['trips'])
    report['duration_seconds'] = round(time.perf_counter()-started,4)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['trip_count','all_match','duration_seconds']}), flush=True)
    return 0 if report['all_match'] else 1

if __name__ == '__main__': raise SystemExit(main())
