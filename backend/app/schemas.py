from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid', allow_inf_nan=False)

class Versioned(Strict):
    schema_version: Literal['1.0']='1.0'

class ErrorInfo(Strict):
    code: str
    message: str
    details: dict=Field(default_factory=dict)
    request_id: str | None=None
    job_id: str | None=None

class ErrorResponse(Versioned):
    error: ErrorInfo

class ModelMetadata(Strict):
    name: Literal['lightgbm']
    version: str | None
    ready: bool
    state: Literal['UNCONFIGURED','INVALID','READY']
    reason: str | None
    feature_schema_version: str
    runtime_pipeline: str

class Health(Versioned):
    status: Literal['alive','ready','degraded']
    model_ready: bool
    model: ModelMetadata
    trip_count: int
    feature_pipeline: str

class TripList(Versioned):
    trips: list[str]

class FeatureAvailability(Strict):
    ready: bool
    reason: str | None

class TripMetadata(Versioned):
    trip_id: str
    observation_count: int
    node_count: int
    arc_count: int
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    nonincreasing_timestamps: int
    large_gaps: int
    features: FeatureAvailability | None=None

class PointGeometry(Strict):
    type: Literal['Point']
    coordinates: tuple[float,float]

class LineGeometry(Strict):
    type: Literal['LineString']
    coordinates: list[tuple[float,float]]

class Feature(Strict):
    type: Literal['Feature']='Feature'
    geometry: PointGeometry | LineGeometry
    properties: dict

class FeatureCollection(Strict):
    type: Literal['FeatureCollection']='FeatureCollection'
    features: list[Feature]

class PreviewGeometry(Strict):
    raw_gps: FeatureCollection
    road_network: FeatureCollection

class RoadPreview(Strict):
    matching_arc_count: int
    returned_arc_count: int
    complete: bool

class TripPreview(Versioned):
    trip_id: str
    metadata: TripMetadata
    geometry: PreviewGeometry
    road_preview: RoadPreview
    warnings: list[str]

class ReconstructionRequest(Strict):
    trip_id: str=Field(pattern=r'^[0-9]{8}$')
    use_hmm: bool=Field(default=True,strict=True)

Stage=Literal['QUEUED','PREPARING_DATA','PREPARING_FEATURES','SCORING_CANDIDATES','DECODING_ROUTE','EVALUATING','COMPLETED','FAILED']
class StageEvent(Strict):
    stage: Stage
    at: str

class JobStatus(Versioned):
    job_id: str
    trip_id: str
    use_hmm: bool
    status: Stage
    created_at: str
    updated_at: str
    duration_ms: float | None
    error: ErrorInfo | None
    stages: list[StageEvent]

class Metrics(Strict):
    top1_accuracy: float | None=Field(default=None,ge=0,le=1)
    route_continuity: float=Field(ge=0,le=1)
    sequence_jaccard: float | None=Field(default=None,ge=0,le=1)
    candidate_miss_rate: float | None=Field(default=None,ge=0,le=1)
    inference_time_ms: float=Field(ge=0)

class Observations(Strict):
    count: int
class Predictions(Strict):
    classifier_arc_sequence: list[int]
    hmm_arc_sequence: list[int]
class ResultGeometry(Strict):
    classifier_route: FeatureCollection
    hmm_route: FeatureCollection
class TransitionBreak(Strict):
    observation_index: int
    reason: Literal['GPS_GAP','NONINCREASING_TIMESTAMP','NO_DIRECTED_CONNECTION']
class Diagnostics(Strict):
    observation_count: int
    decoder_time_ms: float | None
    route_continuity: float
    transition_breaks: list[TransitionBreak]
    policy: str
    gap_threshold_seconds: float | None
class ReconstructionResult(Versioned):
    job_id: str
    trip_id: str
    status: Literal['COMPLETED']
    model: ModelMetadata
    observations: Observations
    geometry: ResultGeometry
    predictions: Predictions
    diagnostics: Diagnostics
    metrics: Metrics
    warnings: list[str]
