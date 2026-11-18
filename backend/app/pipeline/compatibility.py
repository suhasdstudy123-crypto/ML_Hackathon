"""Strict online generation requires authoritative reproduction; demo mode is explicit.

An operator-owned parity report is a compatibility record, not a model artifact
or a public API input. Changing a boolean does not satisfy its evidence checks.
"""
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import scipy
from backend.app.core.errors import AppError

NOTEBOOK_HASH = '9fa739dc905d93e7d46fa31eeb1fad9d4d4916c337ce0bdfec804dec766b1b5d'
ROOT = Path(__file__).resolve().parents[3]
PIPELINE_FILES = ('notebook_features.py', 'reference_alignment.py')
EXTRACTED_FEATURE_HASH = '970ad1803a1d3f25eefa608ba738ff364e5dac1e6e02e0cb95b6f59c8482d330'
DEMO_WARNING = ('Notebook demo mode: candidate identities at geometric ties differ from the supplied '
                'training export in this runtime. Generated features use the unchanged notebook; '
                'exact dataset parity and resulting model accuracy are not verified.')


def runtime_identity():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scipy': scipy.__version__, 'machine': platform.machine(), 'system': platform.system(),
            'byteorder': sys.byteorder}


def pipeline_hashes():
    base = Path(__file__).parent
    return {name: hashlib.sha256((base/name).read_bytes()).hexdigest() for name in PIPELINE_FILES}


class NotebookCompatibility:
    def __init__(self, feature_dir, report_path):
        self.root, self.report_path = Path(feature_dir), Path(report_path)

    def require_ready(self):
        try:
            report = json.loads(self.report_path.read_text())
            if report.get('all_match') is not True:
                raise ValueError('Authoritative feature/alignment parity has not passed')
            if report.get('notebook_sha256') != NOTEBOOK_HASH:
                raise ValueError('Notebook provenance differs')
            if report.get('versions') != runtime_identity() or report.get('pipeline_sha256') != pipeline_hashes():
                raise ValueError('Runtime or extracted pipeline differs from checked parity report')
            checks = json.loads((ROOT/'docs/AUTHORITATIVE_CHECKSUMS.json').read_text())
            checked = report.get('source_artifact_sha256', {})
            for name in ['train_features.parquet','student_test_features.parquet',
                         'train_alignment_diagnostics.parquet','student_test_alignment_diagnostics.parquet']:
                if checked.get(name) != checks[name] or hashlib.sha256((self.root/name).read_bytes()).hexdigest() != checks[name]:
                    raise ValueError('Authoritative artifact differs from checked parity report')
            trips = report.get('trips', [])
            expected_ids = {}
            for split in ['train','student_test']:
                expected_ids[split] = set(pd.read_parquet(self.root/f'{split}_features.parquet', columns=['trip_id']).trip_id.unique())
            keys = [(t.get('split'),t.get('trip_id')) for t in trips]
            expected_keys = {(split,tid) for split,ids in expected_ids.items() for tid in ids}
            if report.get('trip_count') != 90 or len(trips) != 90 or len(keys) != len(set(keys)) or set(keys) != expected_keys:
                raise ValueError('Parity report must cover all 80 training and 10 student-test trips')
            if report.get('all_match') is not True or any(
                t.get('features_match') is not True or t.get('alignment_match') is not True or t.get('labels_match') is not True
                or any(t.get('differences', {}).get(area) != {} for area in ['features','alignment','labels'])
                or set(t.get('raw_sha256',{})) != {'nodes','arcs','track','route'}
                or any(not isinstance(v,str) or len(v)!=64 for v in t.get('raw_sha256',{}).values())
                for t in trips
            ):
                raise ValueError('Notebook candidate identities, features or estimated labels do not reproduce authoritative data')
        except Exception as exc:
            raise AppError('FEATURE_PIPELINE_INCOMPATIBLE',
                'Online notebook generation is blocked: full authoritative dataset parity has not passed in this runtime. Use verified precomputed trip features or reproduce the original export environment.', 503) from exc
        return report

    def require_generation_ready(self, demo_mode=False):
        if not demo_mode:
            return self.require_ready()
        # User-authorized demo policy permits known numerical/tie drift, never source edits.
        source = Path(__file__).parent/'notebook_features.py'
        if hashlib.sha256(source.read_bytes()).hexdigest() != EXTRACTED_FEATURE_HASH:
            raise AppError('FEATURE_PIPELINE_INCOMPATIBLE', 'Notebook source differs from the reviewed demo implementation', 503)
        return {'demo_mode':True, 'warnings':[DEMO_WARNING]}

    def availability(self, demo_mode=False):
        try:
            self.require_generation_ready(demo_mode)
            return {'ready':True,'reason':DEMO_WARNING if demo_mode else None}
        except AppError as exc:
            return {'ready':False,'reason':exc.message}
