"""Extract reviewed definitions without executing a notebook cell.

Only the audited notebook hash is accepted. Runtime imports generated .py files,
never the notebook bootstrap, loader, label attachment, fitting or write cells.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/PathSnap_Dataset_Pipeline_Final_Pointwise.ipynb'
EXPECTED_HASH = '9fa739dc905d93e7d46fa31eeb1fad9d4d4916c337ce0bdfec804dec766b1b5d'
FEATURE_CELLS = {7, 11, 13}
CONSTANTS = {'ALIGNMENT_SIGMA_M', 'ALIGNMENT_MAX_DISTANCE_M', 'ALIGNMENT_MAX_SPEED_MPS',
             'ALIGNMENT_JUMP_SLACK_M', 'K_CANDIDATES', 'NOMINAL_SAMPLING_HZ', 'FEATURE_COLUMNS'}

def extract():
    content = NOTEBOOK.read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_HASH:
        raise ValueError('Notebook changed; source review is required before extraction')
    cells = json.loads(content)['cells']
    constants = []
    for node in ast.parse(''.join(cells[5]['source'])).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in CONSTANTS:
            ast.literal_eval(node.value)  # Reject nonliteral initializers; never evaluate notebook code.
            constants.append(ast.unparse(node))
    header = f'"""Reviewed definitions from the supplied notebook; SHA-256 {EXPECTED_HASH}.\nRegenerate with scripts/extract_notebook_pipeline.py; no algorithm edits here.\n"""\n'
    features = header + '''from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class TripData:
    """Route-blind array adapter. Deliberately has no reference route field."""
    trip_id: str
    nodes: np.ndarray
    arcs: np.ndarray
    track: np.ndarray

'''
    features += '\n'.join(c for c in constants if not c.startswith('ALIGNMENT_')) + '\n\n'
    for index in sorted(FEATURE_CELLS | {18}):
        source = ''.join(cells[index]['source'])
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and (index != 18 or node.name == 'build_trip_table'):
                features += ast.get_source_segment(source, node) + '\n\n'
            elif index == 7 and isinstance(node, ast.Assign):
                ast.literal_eval(node.value)
                features += ast.get_source_segment(source, node) + '\n\n'
    alignment = header + '''from dataclasses import dataclass
import numpy as np
import pandas as pd
from .notebook_features import TripData as FeatureTrip, make_local_projector, point_to_segment_distance, _vehicle_heading_deg, _implied_speed_mps, _time_gaps_s, initial_bearing_deg, angular_difference_deg

@dataclass(frozen=True)
class TripData(FeatureTrip):
    """Protected evaluator-only input, constructed after inference completes."""
    route: np.ndarray

'''
    alignment += '\n'.join(c for c in constants if c.startswith('ALIGNMENT_')) + '\n\n'
    source = ''.join(cells[14]['source'])
    nodes = ast.parse(source).body
    if len(nodes) != 1 or not isinstance(nodes[0], ast.FunctionDef) or nodes[0].name != 'align_gps_to_route':
        raise ValueError('Unexpected alignment source')
    alignment += ast.get_source_segment(source, nodes[0]) + '\n'
    return features, alignment

if __name__ == '__main__':
    features, alignment = extract()
    target = ROOT / 'backend/app/pipeline'
    (target / 'notebook_features.py').write_text(features)
    (target / 'reference_alignment.py').write_text(alignment)
    print('Extracted reviewed definitions only; no notebook cells were executed')
