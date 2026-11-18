from contextlib import asynccontextmanager
import logging
from time import perf_counter
from uuid import uuid4
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from backend.app.config import Settings, ROOT
from backend.app.core.errors import AppError
from backend.app.core.jobs import JobManager
from backend.app.core.logging import configure_logging
from backend.app.models.lightgbm_provider import LightGBMModelProvider
from backend.app.services.trip_service import TripService
from backend.app.services.feature_service import FeatureService
from backend.app.services.inference_service import InferenceService
from backend.app.services.evaluation_service import EvaluationService
from backend.app.schemas import Health, TripList, TripMetadata, TripPreview, ReconstructionRequest, JobStatus, ReconstructionResult, ErrorResponse

configure_logging(); log=logging.getLogger('pathsnap')

def create_app(settings=None, provider=None):
    settings=settings or Settings()
    trips=TripService(settings); features=FeatureService(settings.feature_dir,settings.parity_report,settings.notebook_demo_mode)
    provider=provider or LightGBMModelProvider(settings.manifest_path,settings.feature_dir)
    inference=InferenceService(trips,features,provider); evaluation=EvaluationService(trips)

    @asynccontextmanager
    async def lifespan(app):
        # Startup validates artifacts before requests are accepted.
        if settings.notebook_demo_mode:
            log.warning('notebook_demo_mode_enabled: exact candidate parity is not verified')
        provider.load()
        meta=provider.get_metadata()
        # Printed in the API console so evaluators can see which model is serving requests.
        log.info('model_state',extra={'model_version':meta['version'],'stage':meta['state']})
        app.state.jobs=JobManager(settings,inference,evaluation)
        yield
        app.state.jobs.shutdown()

    app=FastAPI(title='PathSnap — Fleet Route Intelligence',version='1.0.0',lifespan=lifespan,
                responses={400:{'model':ErrorResponse},404:{'model':ErrorResponse},422:{'model':ErrorResponse},503:{'model':ErrorResponse}})
    app.state.settings,app.state.provider,app.state.trips,app.state.features=settings,provider,trips,features
    app.add_middleware(CORSMiddleware,allow_origins=list(settings.origins),allow_credentials=False,
                       allow_methods=['GET','POST'],allow_headers=['Content-Type'],expose_headers=['X-Request-ID'])

    @app.middleware('http')
    async def request_context(request,call_next):
        request.state.request_id=str(uuid4()); started=perf_counter()
        response=await call_next(request)
        response.headers['X-Request-ID']=request.state.request_id
        response.headers['Cache-Control']='no-store'
        log.info('http_request',extra={'request_id':request.state.request_id,'method':request.method,
            'path':request.url.path,'status_code':response.status_code,'duration_ms':(perf_counter()-started)*1000})
        return response

    def error_response(request,exc):
        return JSONResponse(status_code=exc.status,content={'schema_version':'1.0','error':{
            'code':exc.code,'message':exc.message,'details':exc.details,
            'request_id':getattr(request.state,'request_id',None),'job_id':exc.details.get('job_id')}})

    @app.exception_handler(AppError)
    async def app_error(request,exc): return error_response(request,exc)
    @app.exception_handler(RequestValidationError)
    async def validation_error(request,exc):
        # Do not echo submitted values, file paths, arbitrary headers or internal contexts.
        fields=[{'location':list(e['loc']),'type':e['type']} for e in exc.errors()]
        return error_response(request,AppError('INVALID_REQUEST','Request validation failed',422,{'fields':fields}))
    @app.exception_handler(HTTPException)
    async def http_error(request,exc): return error_response(request,AppError('HTTP_ERROR','Requested resource unavailable',exc.status_code))
    @app.exception_handler(Exception)
    async def internal_error(request,exc):
        log.exception('request_failed',extra={'request_id':getattr(request.state,'request_id',None)})
        return error_response(request,AppError('INTERNAL_ERROR','Request failed; inspect server logs',500))

    def health(status):
        ids=trips.ids()
        return {'schema_version':'1.0','status':status,'model_ready':provider.is_ready(),'model':provider.get_metadata(),
                'trip_count':len(ids),'feature_pipeline':'authoritative-notebook-v2; demo mode with known parity differences' if settings.notebook_demo_mode else 'authoritative-notebook-v2; strict parity required'}

    @app.get('/', include_in_schema=False)
    def root(request: Request):
        if 'text/html' in request.headers.get('accept', ''):
            return RedirectResponse('/ui/', status_code=307)
        return health('alive')

    @app.get('/api/v1/health/live',response_model=Health)
    def live(): return health('alive')

    @app.get('/api/v1/health/ready',response_model=Health)
    def ready():
        body=health('ready' if trips.ids() else 'degraded')
        if not trips.ids(): return JSONResponse(status_code=503,content=body)
        return body  # App previews are ready independently from trained model readiness.

    @app.get('/trips',response_model=TripList)
    @app.get('/api/v1/trips',response_model=TripList)
    def list_trips(): return {'schema_version':'1.0','trips':trips.ids()}

    @app.get('/api/v1/trips/{trip_id}',response_model=TripMetadata)
    def metadata(trip_id:str):
        trip=trips.load(trip_id); result=trips.metadata(trip)
        result['features']=features.availability(trip_id, trip.fingerprints)
        return result

    @app.get('/api/v1/trips/{trip_id}/preview',response_model=TripPreview)
    def preview(trip_id:str): return trips.preview(trips.load(trip_id))

    @app.post('/predict_and_evaluate',response_model=JobStatus,status_code=202,deprecated=True)
    @app.post('/api/v1/reconstructions',response_model=JobStatus,status_code=202)
    def reconstruct(body:ReconstructionRequest,request:Request):
        if body.trip_id not in trips.ids(): raise AppError('TRIP_NOT_FOUND','Unknown trip ID',404)
        if not provider.is_ready(): raise AppError('MODEL_NOT_READY','LightGBM model not connected',503)
        # Fail early when model exists but exact authoritative candidate features are absent.
        availability=features.availability(body.trip_id)
        if not availability['ready']: raise AppError('FEATURE_PIPELINE_UNAVAILABLE',availability['reason'],503)
        return app.state.jobs.submit(body.trip_id,body.use_hmm,request.state.request_id)

    @app.get('/api/v1/reconstructions',response_model=list[JobStatus])
    def list_jobs():
        with app.state.jobs.lock:
            app.state.jobs._prune()
            return [app.state.jobs._public(j) for j in app.state.jobs.jobs.values()]

    @app.get('/api/v1/reconstructions/{job_id}',response_model=JobStatus)
    def status(job_id:str): return app.state.jobs.get(job_id)
    @app.get('/api/v1/reconstructions/{job_id}/result',response_model=ReconstructionResult)
    def result(job_id:str): return app.state.jobs.result(job_id)

    @app.get('/ui/runtime-config.js', include_in_schema=False)
    def ui_runtime_config():
        # Browser uses the same HTTPS host as the dashboard; never its own localhost.
        return Response("document.documentElement.dataset.apiMode = 'same-origin';\n",
                        media_type='application/javascript', headers={'Cache-Control':'no-store'})

    ui_dir = ROOT / 'evaluator_ui' / 'dist'
    if ui_dir.is_dir():
        app.mount('/ui', StaticFiles(directory=str(ui_dir), html=True), name='ui')

    return app

app=create_app()
