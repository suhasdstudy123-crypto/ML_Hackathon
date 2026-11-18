# PathSnap — Fleet Route Intelligence

Explore real trip GPS observations and their directed road networks in a Leaflet/OpenStreetMap dashboard. FastAPI runs on Laptop 1; Laptop 2 runs the static dashboard and connects over a configurable LAN URL.

**Current state:** the trained LightGBM model (`lightgbm/`) is connected and all ten reserved evaluator trips reconstruct end to end: online notebook features → LightGBM candidate scores → directed HMM/Viterbi → server-side evaluation (Top-1 accuracy, classifier-only Top-1, route continuity, route-arc precision/recall, candidate miss rate). The backend reproduces the notebook's saved student-test probabilities exactly (max difference 1e-16; Top-1 0.7719). Online candidate ties can differ across runtime environments (demo mode, documented). Top-1 and candidate-miss metrics are measured against accepted estimated notebook alignments.

## Windows quick start (demo)

```powershell
python -m pip install -r requirements-dev.txt -r requirements-model.txt
python scripts\import_dataset.py "<folder or zip with the seven artifacts>"   # once, on Laptop 1
python scripts/validate_model.py                                             # model load + exact parity check
.\start_api.ps1 -UiOrigin http://<Laptop2-IP>:5173                           # Laptop 1 (add -Port 8011 if 8000 is busy)
.\start_ui.ps1 -ApiUrl http://<Laptop1-IP>:8000                              # Laptop 2
```

The model is loaded from `lightgbm/models/model_manifest.json`: the native booster `lightgbm_booster.txt` plus `lightgbm_preprocessing.json`, the exact median-imputer/standard-scaler parameters taken from the trained pipeline by `scripts/export_lightgbm_artifacts.py`. Do **not** substitute `feature_scaler.joblib`: its ~1e-12 parameter drift flips tree splits (probability changes up to 0.165).

```text
Browser (Laptop 2) → versioned HTTP API (Laptop 1)
                         ├─ TripService → track / nodes / arcs → safe preview
                         └─ bounded inference job
                              → notebook candidates/features (no labels; demo parity warning)
                              → LightGBMModelProvider (lightgbm/models, exact preprocessing)
                              → directed Viterbi + classifier sequences
                              → EvaluationService (protected .route, aggregates only)
```

## Setup and startup

Python 3.11+ and Node.js 22+ recommended. The existing scientific requirements are retained; Install `requirements-model.txt` for LightGBM.

```bash
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
npm --prefix evaluator_ui ci
npm --prefix evaluator_ui run typecheck
npm --prefix evaluator_ui run build
```

Run from the repository root. On Laptop 1, set the exact frontend origin for Laptop 2:

```bash
export PATHSNAP_CORS_ORIGINS=http://192.168.1.21:5173
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

On Laptop 2, serve **only the built dashboard directory**, never the repository root:

```bash
python -m http.server 5173 --bind 0.0.0.0 --directory evaluator_ui/dist
```

Open `http://localhost:5173` on Laptop 2. Enter Laptop 1's actual address, such as `http://192.168.1.20:8000`, in **Backend API URL**, then connect. The API URL is saved locally in that browser; a `?api=http%3A%2F%2F192.168.1.20%3A8000` URL parameter can also supply it. The dashboard does not assume that its own host is the backend.

For a single-laptop check, default CORS accepts `http://localhost:5173` and `http://127.0.0.1:5173`. API documentation is at `http://localhost:8000/docs`. Legacy `api_server:app` startup remains supported. See [two-laptop deployment](docs/DEPLOYMENT.md) for Windows commands, firewall checks, origins and troubleshooting.

## Dataset expectations

```text
evaluator_test/evaluator_test/<8-digit-trip-id>/
  <trip-id>.track   # longitude latitude timestamp, ordered observations
  <trip-id>.nodes   # longitude latitude, node row = node ID
  <trip-id>.arcs    # directed start/end node row IDs, arc row = arc ID
  <trip-id>.route   # protected reference sequence; evaluation only

data/authoritative/ # ignored, private on backend laptop
  train_features.parquet
  student_test_features.parquet
  train_alignment_diagnostics.parquet
  student_test_alignment_diagnostics.parquet
  train_diagnostics.csv
  student_test_diagnostics.csv
  feature_scaler.joblib
```

Install the seven supplied artifacts from the original ZIP, preserving bytes:

```bash
python scripts/import_dataset.py "/path/to/outputs (1).zip"
python scripts/verify_artifacts.py
```

The supplied fifteen-feature Parquet schema differs from the preserved historical ten-feature NPZ pipeline. **Do not regenerate or replace either dataset to make them agree.** See [the data contract](docs/DATA_CONTRACT.md), exact [schema inspection](docs/DATA_SCHEMA_INSPECTION.json), and [repository audit](docs/REPOSITORY_AUDIT.md). The ten reserved evaluator trips have no feature rows in the supplied ZIP. Their raw previews do not require feature data or a model; notebook features are generated when genuine inference is requested after model delivery.

## Replacing the model with a newly trained one

1. The supplied notebook now generates evaluator features directly from track/nodes/arcs. Demo mode is enabled by default as requested. Set `PATHSNAP_NOTEBOOK_DEMO_MODE=0` to enforce strict parity, or supply verified precomputed features with raw-file bindings. See [notebook integration](docs/NOTEBOOK_INTEGRATION.md).
2. Train separately on permitted data using the fifteen features in canonical order. Export a native LightGBM Booster or joblib LightGBM classifier; no unrelated classifier is accepted.
3. Copy `models/model_manifest.example.json` to `models/model_manifest.json`. Set the artifact format/path/hash/version, input contract, preprocessing policy/provenance, training reference and `configured=true`.
4. Install `requirements-model.txt`, run `python scripts/validate_model.py --trip 00000023`, then restart the backend and check readiness and trip feature availability.
5. Run real inference/evaluation integration tests after artifacts are available. Existing frontend layers and job monitoring need no rewrite.

No training command is part of backend startup. Do not use a placeholder model or the historical HGB weights. The explicit notebook model/scaler order places `bearing_diff_deg` before `bearing_valid`. The new `notebook-features-v2` manifest uses that order; the earlier provisional v1 model contract is rejected. Physical Parquet order remains unchanged.

## Validation

```bash
python -m pytest -q
npm --prefix evaluator_ui test
npm --prefix evaluator_ui run typecheck
npm --prefix evaluator_ui run build
python scripts/verify_artifacts.py
```

Tests use explicitly named synthetic fixtures only under test directories; production endpoints never return synthetic successes. Legacy root `test_*.py` scripts are preserved exploratory utilities requiring unavailable Windows training paths and are excluded from pytest discovery. [Integration status](docs/INTEGRATION_STATUS.md) lists actual results and untested boundaries.

The original training, EDA, inspection and HGB/HMM experiment scripts, model weights, NPZ caches, figures and HTML maps remain intact and are not imported or served by the production app. The original README is retained at [docs/history/README.original.md](docs/history/README.original.md). `TECHNICAL_REPORT.md` describes historical experiments, not verified current LightGBM capability; its old claim that API labels never leaked was contradicted by the audit and is fixed in the new API. Historical experiment metrics are not displayed as current results.

Further documentation: [API](docs/API_CONTRACT.md) · [Model handoff](docs/MODEL_INTEGRATION.md) · [Deployment](docs/DEPLOYMENT.md) · [Status and limitations](docs/INTEGRATION_STATUS.md).

## One URL with ngrok

Build the dashboard once and serve it from FastAPI, then tunnel the same port:

```bash
npm --prefix evaluator_ui run build
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --workers 1
# In another terminal, unless an existing tunnel already targets port 8000:
ngrok http 8000
```

Share the returned HTTPS URL with `/ui/` appended. Frontend and API use that same origin automatically; the remote laptop needs only a browser, with no local server or API URL entry. A browser request to the bare root redirects to `/ui/`. Keep the API and ngrok running on Laptop 1. This is a public demo link; share only with intended viewers. No raw reference files or model artifacts are mounted as static content. See [unified deployment review](docs/UNIFIED_DEPLOYMENT.md).
