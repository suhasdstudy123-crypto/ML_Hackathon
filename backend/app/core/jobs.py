from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import RLock, Event
from time import monotonic, perf_counter
from uuid import uuid4
import logging
from backend.app.core.errors import AppError

log = logging.getLogger('pathsnap')
TERMINAL = {'COMPLETED', 'FAILED'}

def now(): return datetime.now(timezone.utc).isoformat()

class JobManager:
    """Bounded, single-process, non-durable jobs. Workers are separate from the ASGI event loop."""
    def __init__(self, settings, inference, evaluation):
        self.settings, self.inference, self.evaluation = settings, inference, evaluation
        self.pool = ThreadPoolExecutor(max_workers=settings.max_workers, thread_name_prefix='pathsnap')
        self.lock, self.stopping = RLock(), Event()
        self.jobs = {}

    def _prune(self):
        for jid in list(self.jobs):
            j = self.jobs[jid]
            if j['status'] in TERMINAL and monotonic() - j['_updated'] > self.settings.retention_seconds:
                del self.jobs[jid]

    def submit(self, trip_id, use_hmm, request_id):
        with self.lock:
            self._prune()
            if self.stopping.is_set(): raise AppError('SERVER_SHUTTING_DOWN', 'Server is shutting down', 503)
            for j in self.jobs.values():
                if j['trip_id'] == trip_id and j['use_hmm'] == use_hmm and j['status'] not in TERMINAL:
                    return self._public(j)
            if len(self.jobs) >= self.settings.max_jobs:
                # Evict oldest finished result to bound retained geometry memory.
                finished = [k for k,v in self.jobs.items() if v['status'] in TERMINAL]
                if finished: del self.jobs[finished[0]]
                else: raise AppError('JOB_CAPACITY_REACHED', 'Job capacity reached; retry after an active job finishes', 429)
            jid = str(uuid4())
            j = {'schema_version':'1.0', 'job_id':jid, 'trip_id':trip_id, 'use_hmm':use_hmm, 'status':'QUEUED',
                 'created_at':now(), 'updated_at':now(), 'duration_ms':None, 'error':None,
                 'stages':[{'stage':'QUEUED', 'at':now()}], '_updated':monotonic(), '_result':None, '_request_id':request_id}
            self.jobs[jid] = j
            self.pool.submit(self._run, jid)
            return self._public(j)

    def _public(self,j):
        return {k: ([dict(s) for s in v] if k == 'stages' else v) for k,v in j.items() if not k.startswith('_')}

    def get(self, jid):
        with self.lock:
            self._prune()
            if jid not in self.jobs: raise AppError('JOB_NOT_FOUND', 'Job unavailable or expired; backend restarts clear jobs', 404)
            return self._public(self.jobs[jid])

    def result(self, jid):
        with self.lock:
            job = self.get(jid)
            if job['status'] == 'FAILED':
                raise AppError(job['error']['code'], job['error']['message'], 409, {'job_id':jid})
            if job['status'] != 'COMPLETED': raise AppError('JOB_NOT_COMPLETE', 'Reconstruction is still running', 409, {'job_id':jid})
            return self.jobs[jid]['_result']

    def _run(self,jid):
        started = perf_counter()
        deadline = monotonic()+self.settings.timeout_seconds
        with self.lock: j = self.jobs[jid]
        def check():
            if self.stopping.is_set(): raise AppError('SERVER_SHUTTING_DOWN', 'Inference stopped during shutdown', 503)
            if monotonic() > deadline: raise AppError('INFERENCE_TIMEOUT', 'Inference exceeded configured execution deadline', 504)
        def stage(value):
            check()
            with self.lock:
                j['status'], j['updated_at'], j['_updated'] = value, now(), monotonic()
                j['stages'].append({'stage':value,'at':j['updated_at']})
            log.info('job_stage',extra={'request_id':j['_request_id'],'job_id':jid,'trip_id':j['trip_id'],'stage':value})
        try:
            trip, seq, mapping, result = self.inference.run(j['trip_id'],j['use_hmm'],stage,check)
            check(); stage('EVALUATING')
            metrics,warnings = self.evaluation.evaluate(trip,seq,mapping)
            check()
            result['metrics'].update(metrics); result['warnings'].extend(warnings); result['job_id'] = jid
            from backend.app.schemas import ReconstructionResult
            result = ReconstructionResult.model_validate(result).model_dump()
            with self.lock: j['_result'] = result
            stage('COMPLETED')
        except Exception as exc:
            error = exc if isinstance(exc,AppError) else AppError('INTERNAL_ERROR','Reconstruction failed; inspect server logs',500)
            with self.lock:
                j['error']={'code':error.code,'message':error.message,'details':{},'request_id':j['_request_id'],'job_id':jid}
                j['status'], j['updated_at'], j['_updated'] = 'FAILED',now(),monotonic()
                j['stages'].append({'stage':'FAILED','at':j['updated_at']})
                j['_result'] = None
            log.exception('job_failed',extra={'request_id':j['_request_id'],'job_id':jid,'trip_id':j['trip_id'],'error_code':error.code})
        finally:
            with self.lock: j['duration_ms'] = (perf_counter()-started)*1000
            log.info('job_finished',extra={'request_id':j['_request_id'],'job_id':jid,'trip_id':j['trip_id'],'stage':j['status'],'duration_ms':j['duration_ms']})

    def shutdown(self):
        self.stopping.set()
        # Queued work observes stop before data loading; active work checks between stages and decoder observations.
        self.pool.shutdown(wait=True, cancel_futures=False)
