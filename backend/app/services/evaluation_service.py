import logging
import numpy as np
import pandas as pd
from backend.app.core.errors import AppError
from backend.app.pipeline.reference_alignment import TripData as ReferenceTrip, align_gps_to_route

log = logging.getLogger('pathsnap')

class EvaluationService:
    def __init__(self, trip_service): self.trips = trip_service

    def _reference(self, trip):
        # This is the sole production reader of reference files, called after inference returns.
        p = self.trips.path(trip.trip_id, 'route')
        try:
            frame = pd.read_csv(p, sep=r'\s+', header=None)
            values = frame.to_numpy(dtype=float)
            if frame.shape[1] != 1 or not len(frame) or not np.isfinite(values).all() or ((values < 0) | (values >= len(trip.arcs)) | (values != np.floor(values))).any():
                raise ValueError('Invalid reference shape or IDs')
            return values[:, 0].astype(np.int64)
        except Exception as exc:
            raise AppError('INVALID_REFERENCE_DATA', 'Protected evaluation data could not be validated', 500) from exc

    def evaluate(self, trip, predicted, mapping, classifier=None):
        """Score a finished prediction against the protected reference route.

        predicted / classifier: one predicted arc ID per GPS observation (HMM or classifier-only).
        mapping: the candidate rows (point_idx, candidate_edge_id) that were scored.
        Only aggregate numbers leave this method; reference arcs are never returned.
        """
        try: route = self._reference(trip)
        except AppError as exc:
            if exc.code == 'TRIP_FILE_MISSING': return {}, ['Reference unavailable; evaluation omitted']
            raise
        reference, pred = set(route.tolist()), set(int(a) for a in predicted)
        metrics = {'sequence_jaccard': len(pred & reference) / len(pred | reference),
                   'route_arc_precision': len(pred & reference) / len(pred) if pred else None,
                   'route_arc_recall': len(pred & reference) / len(reference)}
        warnings = ['Top-1 accuracy and candidate miss rate use accepted estimated notebook alignments, not independently verified per-observation ground truth.']
        try:
            # Same notebook alignment that produced the training labels: each GPS point gets the
            # reference arc it most likely lies on. Alignment is an estimated label;
            # parity and candidate tie behavior can vary across runtime environments.
            aligned = align_gps_to_route(ReferenceTrip(trip.trip_id, trip.nodes[['lon', 'lat']].to_numpy(),
                                                       trip.arcs[['u', 'v']].to_numpy(), trip.track[['lon', 'lat', 'time']].to_numpy(), route))
            aligned = aligned.sort_values('point_idx')
            accepted = aligned.alignment_status.eq('accepted').to_numpy()
            truth = aligned.aligned_edge_id.to_numpy()
            if len(truth) != len(predicted) or not accepted.any():
                raise ValueError('Alignment does not cover the predicted observations')
            hit = np.asarray(predicted, dtype=np.int64) == truth
            metrics['top1_accuracy'] = float(hit[accepted].mean())
            metrics['evaluated_points'] = int(accepted.sum())
            if classifier is not None:
                metrics['classifier_top1_accuracy'] = float((np.asarray(classifier, dtype=np.int64) == truth)[accepted].mean())
            # Candidate miss: the reference arc was not among the five generated candidates.
            candidates = mapping.groupby('point_idx').candidate_edge_id.apply(set)
            covered = np.array([truth[i] in candidates.get(i, set()) for i in range(len(truth))])
            metrics['candidate_miss_rate'] = float(1 - covered[accepted].mean())
        except Exception:
            log.exception('alignment_failed', extra={'trip_id': trip.trip_id})
            warnings.append('Top-1 accuracy unavailable: reference alignment failed for this trip')
        log.info('evaluation_complete', extra={'trip_id': trip.trip_id, 'metrics': {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}})
        return metrics, warnings
