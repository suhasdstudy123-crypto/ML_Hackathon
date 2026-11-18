from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd
from backend.app.core.errors import AppError

@dataclass(frozen=True)
class TripInputs:
    trip_id: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    fingerprints: dict

class TripService:
    def __init__(self, settings): self.settings = settings

    def ids(self):
        root = self.settings.trip_dir
        if not root.is_dir(): return []
        return sorted(p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r'[0-9]{8}', p.name) and not p.is_symlink() and all((p / f'{p.name}.{ext}').is_file() for ext in ['track', 'nodes', 'arcs']))

    def path(self, trip_id, ext):
        if not re.fullmatch(r'[0-9]{8}', trip_id) or trip_id not in self.ids():
            raise AppError('TRIP_NOT_FOUND', 'Unknown trip ID', 404)
        p = self.settings.trip_dir / trip_id / f'{trip_id}.{ext}'
        if p.is_symlink() or not p.resolve().is_relative_to(self.settings.trip_dir.resolve()):
            raise AppError('INVALID_TRIP_PATH', 'Trip file is not within the configured dataset', 400)
        if not p.is_file(): raise AppError('TRIP_FILE_MISSING', 'Required trip file is unavailable', 422)
        if p.stat().st_size > self.settings.max_file_bytes:
            raise AppError('TRIP_LIMIT_EXCEEDED', 'Trip file exceeds configured size limit', 413)
        return p

    def load(self, trip_id):
        # Intentionally never reads .route. Only EvaluationService has reference access.
        frames, hashes = {}, {}
        for ext, names in [('track', ['lon', 'lat', 'time']), ('nodes', ['lon', 'lat']), ('arcs', ['u', 'v'])]:
            p = self.path(trip_id, ext)
            try:
                values = pd.read_csv(p, sep=r'\s+', header=None)
                if values.shape[1] != len(names) or values.empty: raise ValueError('Invalid shape')
                values.columns = names
                if not np.isfinite(values.to_numpy(dtype=float)).all(): raise ValueError('Nonfinite data')
                frames[ext] = values
                hashes[ext] = hashlib.sha256(p.read_bytes()).hexdigest()
            except Exception as exc:
                raise AppError('CORRUPT_TRIP_DATA', 'Trip data contains invalid numeric values or columns') from exc
        tr, nd, ar = frames['track'], frames['nodes'], frames['arcs']
        for d in [tr, nd]:
            if (d.lon.abs() > 180).any() or (d.lat.abs() > 90).any():
                raise AppError('INVALID_COORDINATES', 'Trip coordinates are outside valid longitude/latitude ranges')
        if len(tr) > self.settings.max_observations or len(ar) > self.settings.max_arcs:
            raise AppError('TRIP_LIMIT_EXCEEDED', 'Trip exceeds configured observation or graph limits', 413)
        arr = ar.to_numpy(dtype=float)
        if ((arr < 0) | (arr >= len(nd)) | (arr != np.floor(arr))).any():
            raise AppError('INVALID_ARC_INDICES', 'Directed arc endpoint does not refer to a valid node')
        return TripInputs(trip_id, tr.astype(float), nd.astype(float), ar.astype('int64'), hashes)

    def metadata(self, trip):
        times = trip.track.time.to_numpy()
        dt = np.diff(times)
        return {'schema_version': '1.0', 'trip_id': trip.trip_id, 'observation_count': len(trip.track),
                'node_count': len(trip.nodes), 'arc_count': len(trip.arcs),
                'start_timestamp': float(times[0]), 'end_timestamp': float(times[-1]),
                'duration_seconds': float(times[-1] - times[0]),
                'nonincreasing_timestamps': int((dt <= 0).sum()), 'large_gaps': int((dt > 30).sum())}

    def preview(self, trip):
        nd, ar, tr = trip.nodes[['lon', 'lat']].to_numpy(), trip.arcs.to_numpy(), trip.track
        # Viewport geometry only; full directed graph is always used for inference.
        lo = tr[['lon','lat']].min().to_numpy() - .005
        hi = tr[['lon','lat']].max().to_numpy() + .005
        starts, ends = nd[ar[:,0]], nd[ar[:,1]]
        mask = ((np.maximum(starts, ends) >= lo) & (np.minimum(starts, ends) <= hi)).all(axis=1)
        ids = np.flatnonzero(mask)
        limit = 6000
        warnings = []
        if len(ids) > limit: warnings.append(f'Road preview limited to {limit} of {len(ids)} viewport arcs; inference graph remains complete')
        gps = collection([{'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [float(row.lon), float(row.lat)]},
                            'properties': {'observation_index': i, 'timestamp': float(row.time)}} for i,row in enumerate(tr.itertuples(index=False))])
        return {'schema_version': '1.0', 'trip_id': trip.trip_id, 'metadata': self.metadata(trip),
                'geometry': {'raw_gps': gps, 'road_network': arc_geometry(trip, ids[:limit])},
                'road_preview': {'matching_arc_count': len(ids), 'returned_arc_count': min(len(ids), limit), 'complete': len(ids) <= limit},
                'warnings': warnings}

def collection(features): return {'type': 'FeatureCollection', 'features': features}

def arc_geometry(trip, sequence):
    nodes = trip.nodes[['lon', 'lat']].to_numpy()
    arcs = trip.arcs.to_numpy()
    result = []
    for index, arc in enumerate(sequence):
        if arc < 0 or arc >= len(arcs): raise AppError('INVALID_CANDIDATE_MAPPING', 'Predicted arc outside trip graph')
        u, v = arcs[arc]
        result.append({'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': [nodes[u].tolist(), nodes[v].tolist()]},
                       'properties': {'arc_id': int(arc), 'sequence_index': index, 'trip_id': trip.trip_id}})
    return collection(result)
