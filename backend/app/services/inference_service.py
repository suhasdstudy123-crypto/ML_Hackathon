import logging
import time
from backend.app.models.contract import group_scores
from backend.app.decoding.hmm_adapter import decode, continuity
from .trip_service import arc_geometry, collection

log = logging.getLogger('pathsnap')

class InferenceService:
    def __init__(self, trip_service, feature_service, provider):
        self.trips, self.features, self.provider = trip_service, feature_service, provider

    def run(self, trip_id, use_hmm, stage, check_deadline):
        start = time.perf_counter()
        stage('PREPARING_DATA')
        trip = self.trips.load(trip_id); check_deadline()
        stage('PREPARING_FEATURES')
        X, mapping = self.features.prepare(trip); check_deadline()
        stage('SCORING_CANDIDATES')
        scores = self.provider.predict_candidate_scores(X); check_deadline()
        groups = group_scores(X, mapping, scores, len(trip.track), len(trip.arcs))
        log.info('candidate_scoring_complete', extra={'trip_id': trip_id, 'candidate_count': len(X), 'model_version': self.provider.get_metadata()['version']})
        if use_hmm:
            stage('DECODING_ROUTE')
            decoded = decode(groups, trip.arcs, trip.track.time.to_numpy(), check_deadline)
        else:
            seq = [g[0]['arc_id'] for g in groups]
            decoded = {'classifier_arcs': seq, 'hmm_arcs': [], 'diagnostics': {'observation_count': len(seq), 'decoder_time_ms': None,
                'route_continuity': continuity(seq, trip.arcs), 'transition_breaks': [], 'policy': 'classifier_only', 'gap_threshold_seconds': None}}
        check_deadline()
        log.info('decoder_complete', extra={'trip_id': trip_id, 'duration_ms': (time.perf_counter()-start)*1000})
        seq = decoded['hmm_arcs'] if use_hmm else decoded['classifier_arcs']
        # Evaluation is a separate caller after this method returns. No reference data enters here.
        result = {'schema_version': '1.0', 'trip_id': trip_id, 'status': 'COMPLETED', 'model': self.provider.get_metadata(),
                  'observations': {'count': len(trip.track)},
                  'geometry': {'classifier_route': arc_geometry(trip, decoded['classifier_arcs']),
                               'hmm_route': arc_geometry(trip, decoded['hmm_arcs']) if use_hmm else collection([])},
                  'predictions': {'classifier_arc_sequence': decoded['classifier_arcs'], 'hmm_arc_sequence': decoded['hmm_arcs']},
                  'diagnostics': decoded['diagnostics'],
                  'metrics': {'top1_accuracy': None, 'route_continuity': decoded['diagnostics']['route_continuity'], 'sequence_jaccard': None,
                              'candidate_miss_rate': None, 'inference_time_ms': (time.perf_counter()-start)*1000},
                  'warnings': ['Directed run breaks are reported explicitly; geometry contains individual predicted arcs without invented connecting roads'] if decoded['diagnostics']['transition_breaks'] else []}
        return trip, seq, mapping, result
