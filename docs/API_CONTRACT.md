> Notebook update: the explicit model feature order is versioned `notebook-features-v2`. User-authorized demo mode enables the supplied route-blind notebook generator with a warning about candidate tie drift. Set PATHSNAP_NOTEBOOK_DEMO_MODE=0 for strict parity. Model readiness remains independent and false without LightGBM. See NOTEBOOK_INTEGRATION.md.

# PathSnap API v1

JSON responses carry `schema_version: "1.0"`. Exceptions use the same version. Runtime Pydantic response schemas and generated `/openapi.json` describe the exact contract. Frontend JSDoc/TypeScript declarations live in `evaluator_ui/js/types.d.ts`; the typed API client rejects other versions.

| Method | Endpoint | Meaning |
|---|---|---|
| GET | /api/v1/health/live | 200 process alive; includes model state separately |
| GET | /api/v1/health/ready | 200 when permitted trip previews available, 503 degraded when no raw trips; model_ready remains independent |
| GET | /api/v1/trips | Permitted eight-digit trip IDs, including all ten reserved evaluator trips in this checkout |
| GET | /api/v1/trips/{trip_id} | GPS/node/arc counts, start/end/duration, timestamp issues and feature source availability |
| GET | /api/v1/trips/{trip_id}/preview | Complete permitted GPS observations and bounded viewport road geometry, no reference labels |
| POST | /api/v1/reconstructions | Validate request, check model/features, return 202 actual job status; 503 MODEL_NOT_READY before job creation when model absent |
| GET | /api/v1/reconstructions | Retained current-session jobs for overview/reconnection |
| GET | /api/v1/reconstructions/{job_id} | Job state, timestamps, actual stage history, timing/error |
| GET | /api/v1/reconstructions/{job_id}/result | Completed prediction-only geometry and aggregate metrics; 409 incomplete/failed, 404 expired/restarted/unknown |

Legacy GET `/` returns live health; GET `/trips` aliases the versioned trip list. Deprecated POST `/predict_and_evaluate` shares reconstruction business logic and now returns an asynchronous job (202). Its former unsafe synchronous response containing hidden labels is intentionally incompatible and removed. No public experiments, dataset, static reference maps, upload, file download or debug endpoints are exposed. Model Lab is omitted until historical results are reproduced/verified under consistent definitions.

Request:

```json
{"trip_id":"00000023","use_hmm":true}
```

Only these fields are allowed; use_hmm is a strict boolean. Malformed IDs/requests receive 422, unknown well-formed trip IDs 404. Path traversal and out-of-directory/symlink files are rejected. No arbitrary file paths are accepted from clients.

Current honest response, absent trained model:

```json
{
  "schema_version":"1.0",
  "error":{
    "code":"MODEL_NOT_READY",
    "message":"LightGBM model not connected",
    "details":{},
    "request_id":"generated-request-uuid",
    "job_id":null
  }
}
```

Every response includes `X-Request-ID`; no submitted header value is trusted as that ID. Detailed tracebacks stay in backend logs. Validation errors show field locations/types without echoing user inputs. Stable errors include MODEL_NOT_READY, MODEL_ARTIFACT_MISSING, INVALID_MODEL_ARTIFACT, FEATURE_PIPELINE_UNAVAILABLE, FEATURE_BINDING_MISMATCH, FEATURE_PIPELINE_INCOMPATIBLE, INVALID_FEATURE_SCHEMA, INVALID_FEATURE_VALUES, EMPTY_CANDIDATE_GROUP, INVALID_CANDIDATE_MAPPING, INVALID_MODEL_OUTPUT, TRIP_NOT_FOUND, TRIP_FILE_MISSING, CORRUPT_TRIP_DATA, INVALID_COORDINATES, INVALID_ARC_INDICES, TRIP_LIMIT_EXCEEDED, JOB_CAPACITY_REACHED, JOB_NOT_FOUND, JOB_NOT_COMPLETE, INFERENCE_TIMEOUT, SERVER_SHUTTING_DOWN and INTERNAL_ERROR. Loader failure specifics are recorded in backend logs; reconstruction remains MODEL_NOT_READY while state=INVALID.

Stages, only when actually executed:

```text
QUEUED → PREPARING_DATA → PREPARING_FEATURES → GENERATING_CANDIDATES (online only) → SCORING_CANDIDATES
       → DECODING_ROUTE (if use_hmm) → EVALUATING → COMPLETED
Any executing stage → FAILED
```

GENERATING_CANDIDATES is emitted only when the supplied notebook actually generates features; precomputed feature jobs omit it. Each stage includes an ISO-8601 UTC timestamp. No completion percentage is invented. A matching active trip+options request returns the existing job ID. Capacity exhaustion returns 429; completed records can expire after retention or be evicted at capacity. The UI polls at 1-second intervals, retries transient GET network failures three times with backoff, stops after 300 checks and can resume a stored actual job on reconnect. POST is not automatically retried. Backend execution is independent of browser connection. No SSE/WebSocket is needed for this MVP.

Completed results (contract, **not a claimed real inference example**): job_id/trip_id/status/model; observations.count; geometry.classifier_route/hmm_route as FeatureCollections; predictions.classifier_arc_sequence/hmm_arc_sequence; diagnostics with decoder policy/runtime/continuity/explicit transition breaks; metrics and warnings. Raw GPS and roads are retrieved separately from preview, not duplicated in results. Classifier-only mode returns an empty HMM geometry/sequence and null decoder runtime.

Metric units:

- Accuracy, continuity, Jaccard and candidate miss rate are **fractions [0,1]**, displayed as percentages by the browser.
- `top1_accuracy` and `candidate_miss_rate` remain null pending acceptance of the notebook alignment parity limitation. Missing labels are not counted as negative or wrong by inference.
- `sequence_jaccard` is unique predicted versus protected reference arc-set intersection/union. Despite its API name, it ignores ordering and repetition. It is null without reference data. It is not edit distance or observation accuracy.
- `route_continuity` uses equal/adjacent directed consecutive predicted arcs over all consecutive observations; a single observation has continuity 1.
- `inference_time_ms` measures trip preparation through scoring/decoding/geometry; excludes post-inference evaluation. Job duration includes evaluation. Decoder timing uses milliseconds; metadata duration uses seconds.
- Coordinates are WGS84 GeoJSON `[longitude, latitude]`. GPS index and array order match original track order. Route LineStrings are individual directed segments with arc_id/sequence_index/trip_id; repeated arcs are retained.
- Empty candidate groups fail explicitly; no `-1` arc indexing or fabricated route fallback. Disconnected valid runs are retained and reported as transition breaks.

Preview returns all GPS observations (up to configured raw input limits). Road viewport is GPS bounds plus .005 degrees, capped at 6,000 arcs with `road_preview.complete=false`, counts and a warning when limited. The browser draws in batches on canvas. Only visualization is limited; inference uses the complete original graph and all observations.

Hidden reference arc IDs, geometries, labels, alignment status and per-observation correctness are never serialized. Evaluation can read `.route` only after InferenceService returns, and reports aggregates. Request/job JSON logs prove real execution stages without printing hidden labels. Logs are operator-side and not accessible via the API.
