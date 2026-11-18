"""Strict directed-adjacency Viterbi. Disconnected runs are returned as explicit breaks."""
import time
import numpy as np
from backend.app.core.errors import AppError

def continuity(sequence, arcs):
    if len(sequence) < 2: return 1.0
    return sum(a == b or int(arcs.iloc[a].v) == int(arcs.iloc[b].u) for a,b in zip(sequence, sequence[1:])) / (len(sequence)-1)

def decode(groups, arcs, timestamps, check_deadline=lambda: None, gap_seconds=30.0):
    started = time.perf_counter()
    if not groups or len(groups) != len(timestamps) or any(not g for g in groups):
        raise AppError('EMPTY_CANDIDATE_GROUP', 'Decoder requires candidates for every observation')
    for g in groups:
        for c in g:
            if not 0 <= c['arc_id'] < len(arcs) or not np.isfinite(c['probability']) or not 0 <= c['probability'] <= 1:
                raise AppError('INVALID_DECODER_INPUT', 'Invalid arc ID or probability')
    u, v = arcs.u.to_numpy(), arcs.v.to_numpy()
    classifier = [max(g, key=lambda c:c['probability'])['arc_id'] for g in groups]
    decoded, breaks, tables, pointers = [], [], [], []
    segment_start = 0

    def flush():
        if not tables: return
        idx = int(np.argmax(tables[-1])); seq = []
        for local in range(len(tables)-1, -1, -1):
            seq.append(groups[segment_start + local][idx]['arc_id'])
            idx = pointers[local][idx]
        decoded.extend(reversed(seq))

    for t, current in enumerate(groups):
        check_deadline()
        # Independent binary probabilities are discriminative emission potentials.
        # Per-group normalization adds the same log constant to all paths and cannot alter argmax.
        emission = np.log(np.maximum([c['probability'] for c in current], 1e-12))
        reason = None
        if t:
            dt = timestamps[t]-timestamps[t-1]
            if dt <= 0: reason = 'NONINCREASING_TIMESTAMP'
            elif dt > gap_seconds: reason = 'GPS_GAP'
        if tables and reason is None:
            scores, bp = [], []
            for j,c in enumerate(current):
                options = np.array([tables[-1][k] if p['arc_id'] == c['arc_id'] or v[p['arc_id']] == u[c['arc_id']] else -np.inf for k,p in enumerate(groups[t-1])])
                best = int(np.argmax(options))
                scores.append(options[best] + emission[j]); bp.append(best)
            if not np.isfinite(scores).any(): reason = 'NO_DIRECTED_CONNECTION'
        if reason is not None:
            flush(); tables, pointers = [], []
            segment_start = t
            breaks.append({'observation_index': t, 'reason': reason})
        if not tables:
            tables.append(emission); pointers.append([-1]*len(current))
        else:
            tables.append(np.asarray(scores)); pointers.append(bp)
    flush()
    return {'classifier_arcs': classifier, 'hmm_arcs': decoded,
            'diagnostics': {'observation_count': len(groups), 'decoder_time_ms': (time.perf_counter()-started)*1000,
                            'route_continuity': continuity(decoded, arcs), 'transition_breaks': breaks,
                            'policy': 'strict_directed_adjacency_v1', 'gap_threshold_seconds': gap_seconds}}
