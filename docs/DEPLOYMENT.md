# Two-laptop LAN deployment

Laptop 1 hosts Python/FastAPI, private trip data, feature artifacts, model and evaluation. Laptop 2 needs a browser plus a static HTTP server for the dashboard. REST with bounded polling is sufficient; no WebSocket/SSE dependencies are required. Browser disconnection does not cancel backend inference.

## Laptop 1

Use a dedicated Python environment from one interpreter. During development this machine had conflicting framework/Anaconda Python installations; the selected framework Python 3.13 environment passed tests. A different Anaconda PyArrow stack aborted on Parquet reads. Create/activate a virtual environment consistently and check `python -c "import sys; print(sys.executable)"` if results differ between shells.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/import_dataset.py "/path/to/outputs (1).zip"
python scripts/verify_artifacts.py
export PATHSNAP_CORS_ORIGINS=http://192.168.1.21:5173
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

Windows PowerShell equivalent after creating the environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
$env:PATHSNAP_CORS_ORIGINS="http://192.168.1.21:5173"
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

Use `ipconfig` on Windows, `ipconfig getifaddr en0` on macOS Wi-Fi, or `ip -4 addr` on Linux to find Laptop 1's LAN IPv4 address. `0.0.0.0` is the bind address, **not** a browser destination. Check `http://<Laptop1-IP>:8000/api/v1/health/ready`. Without a model, previews can be ready while `model_ready=false`; this is expected.

## Laptop 2

Copy/clone the repository, then build the frontend (no data required on Laptop 2):

```bash
npm --prefix evaluator_ui ci
npm --prefix evaluator_ui run typecheck
npm --prefix evaluator_ui run build
python -m http.server 5173 --bind 0.0.0.0 --directory evaluator_ui/dist
```

Open the dashboard at `http://localhost:5173` on Laptop 2, or its LAN URL if desired. Configure CORS on Laptop 1 for the **exact URL origin used in that browser**, including scheme, hostname/IP and port. If the dashboard is opened using `http://localhost:5173`, use that as an explicitly trusted frontend origin; the browser's Origin refers to Laptop 2's dashboard URL, not to the backend address. Multiple trusted origins are comma-separated. Never add `null` to support file:// pages.

Enter `http://<Laptop1-IP>:8000` in Backend API URL. Click connect, select a trip and verify GPS/roads appear. Backend URL is saved to browser local storage and can be changed without rebuilding. Reconstruction remains disabled until both model and selected-trip features are available. No sample route/metric is substituted.

## Environment variables

Paths default to repository-relative locations resolved from source, rather than Windows user directories. Explicit relative environment paths resolve from process working directory; run commands from repository root or use absolute paths. `.env.example` is a template; export values explicitly or load with your process manager. No implicit dotenv dependency is used.

| Variable | Default / meaning |
|---|---|
| PATHSNAP_TRIP_DIR | `<repo>/evaluator_test/evaluator_test` |
| PATHSNAP_FEATURE_DIR | `<repo>/data/authoritative` |
| PATHSNAP_MODEL_MANIFEST | `<repo>/models/model_manifest.json` (absent by design) |
| PATHSNAP_CORS_ORIGINS | `http://localhost:5173,http://127.0.0.1:5173`; replace/add exact Laptop 2 browser origin |
| PATHSNAP_MAX_WORKERS | 1 inference thread; range 1–4; use one ASGI process |
| PATHSNAP_MAX_JOBS | 16 active+retained jobs, at most 128; cannot be below worker count |
| PATHSNAP_JOB_RETENTION_SECONDS | 3600; finished jobs expire lazily when requested/listed or new jobs arrive |
| PATHSNAP_JOB_TIMEOUT_SECONDS | 120 execution seconds, excluding queue wait; cooperative deadline |

Hard raw-input limits: 25,000 observations, 400,000 directed arcs, 64 MiB per raw file. Visualization draws all GPS samples and up to 6,000 viewport road arcs, with explicit truncation metadata. Complete graph and observations remain unchanged for inference.

## Jobs and resource limits

Inference uses a bounded ThreadPoolExecutor away from the async event loop; native model inference uses one LightGBM thread per job. Duplicate active trip/options requests share a job. Capacity counts queued/active/retained jobs; oldest finished results can be evicted at capacity, otherwise new work receives 429. Terminal jobs contain prediction geometry only. Job progress has timestamps, no invented percentages.

In-memory state disappears on process restart. Multiple Uvicorn workers would have separate state; use `--workers 1`. This is not a durable distributed task queue. Execution deadlines are checked between preparation/scoring/evaluation stages and per Viterbi observation. A running native predict call or disk read **cannot be forcibly interrupted by a thread deadline**; its worker remains occupied until it returns, then the overdue job fails without a result. Shutdown stops new jobs, asks work to stop cooperatively, and waits for active native calls. If actual trained-model benchmarks establish a need for hard process deadlines/durable state, adopt process workers/shared storage then; do not imply those guarantees now.

## Network checklist and troubleshooting

1. Both laptops must be on the same reachable LAN. Guest Wi-Fi/client isolation or a VPN may prevent peer traffic even when both have internet.
2. Check backend health locally, then from Laptop 2 using Laptop 1's IPv4 address. Confirm the server binds `0.0.0.0`, not only loopback.
3. If a port is already in use, select another (for example 8011) and update the browser API field. Do not terminate unrelated processes.
4. Allow only the chosen backend TCP port on the trusted/private firewall profile. Do not disable the firewall. Confirm campus/router policy allows peer traffic.
5. If health opens directly but the dashboard request fails, check exact CORS scheme/host/port. API CORS has explicit trusted origins and no credentialed wildcard. See [FastAPI CORS documentation](https://fastapi.tiangolo.com/tutorial/cors/).
6. Serve the LAN dashboard over HTTP when the backend is HTTP. An HTTPS frontend cannot fetch an HTTP backend without mixed-content errors. The client gives explicit guidance rather than bypassing browser security.
7. If internet access fails, Leaflet CDN/fonts/OSM background may be unavailable. No offline basemap is bundled. Tile failures show a message; if Leaflet itself fails to load, reload after restoring access. Vendor Leaflet locally for a truly offline demo.
8. Visible OpenStreetMap attribution is retained; the app uses the standard HTTPS tile endpoint and no bulk prefetch. Respect the [OSM tile policy](https://operations.osmfoundation.org/policies/tiles/); for high volume/offline use, configure a suitable hosted/self-hosted OSM tile service in `map.js`.
9. Reconnect through the connection button after changing URLs. Health is checked every 15 seconds when idle. Job polling retries transient GET failures and can resume a saved actual job ID. If the backend restarted or retention expired, JOB_NOT_FOUND explains that the previous job is gone.
10. `MODEL_NOT_READY` is expected until model delivery. `FEATURE_PIPELINE_UNAVAILABLE` requires the original authoritative generator or exact feature export for the selected trip. Do not use historical HGB models/NPZ features as fallback.

Use `python -m uvicorn ... 2>&1 | tee pathsnap-backend.log` on a trusted operator terminal to retain JSON execution logs, then keep that log outside the public static directory. Logs contain request/job/trip IDs, stage changes, candidate counts, model version, duration, decode/evaluation completion and failures, without hidden labels. Retain according to the evaluator's needs; no public log endpoint exists.

Plain HTTP with no authentication is appropriate only for this trusted hackathon LAN demonstration. Internet-facing deployment requires TLS, authentication/authorization, rate limits, protected data storage, restricted docs/admin surfaces and durable job state. Do not expose the repository root or private evaluator files through a static server.

## Verification performed versus on-site checks

Automated API tests covered all ten real trips. Browser tests exercised actual HTTP communication between dashboard port 5173 and API port 8011 through both loopback and the laptop's LAN IPv4 address, including reconnect, layer controls and model-pending states. Desktop (1366×980) and narrow (390×844) layouts were inspected. Both servers bound 0.0.0.0. This verifies same-machine LAN addressing; **physical second-laptop reachability and its firewall/CORS setup were not tested**. Run the checklist on the two actual laptops before demonstration.
