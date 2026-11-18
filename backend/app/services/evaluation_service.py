import logging
import numpy as np
import pandas as pd
from backend.app.core.errors import AppError

log = logging.getLogger('pathsnap')

class EvaluationService:
    def __init__(self, trip_service): self.trips = trip_service

    def evaluate(self, trip, predicted, mapping):
        # This is the sole production reader of reference files, called after inference returns.
        try: p = self.trips.path(trip.trip_id, 'route')
        except AppError as exc:
            if exc.code == 'TRIP_FILE_MISSING': return {}, ['Reference unavailable; evaluation omitted']
            raise
        try:
            frame = pd.read_csv(p, sep=r'\s+', header=None)
            values = frame.to_numpy(dtype=float)
            if frame.shape[1] != 1 or not len(frame) or not np.isfinite(values).all() or ((values < 0) | (values >= len(trip.arcs)) | (values != np.floor(values))).any():
                raise ValueError('Invalid reference shape or IDs')
            reference = set(values[:,0].astype(int))
        except Exception as exc:
            raise AppError('INVALID_REFERENCE_DATA', 'Protected evaluation data could not be validated', 500) from exc
        pred = set(predicted)
        metrics = {'sequence_jaccard': len(pred & reference)/len(pred | reference)}
        # Supplied notebook alignment does not yet fully reproduce authoritative diagnostics.
        # Do not substitute a drifting pseudo-label estimate for verified evaluation.
        log.info('evaluation_complete', extra={'trip_id': trip.trip_id})
        return metrics, ['Top-1 accuracy and candidate miss rate unavailable until notebook alignment parity is verified; Jaccard compares unique arc sets']
