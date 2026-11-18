# Unified deployment review — 18 November 2026

Reviewed teammate commit `64db825895c0e8082be2266291ae41eb7c8eeb9c`. The supplied native LightGBM booster, exact JSON median/scaler parameters, explicit positional feature binding, bounded jobs and post-inference aggregate evaluation are integrated. This deployment does not train a model or substitute outputs.

## Findings and changes

- The existing local server was still running the earlier model-unconfigured code. Updating a checkout does not reload a non-reloading Uvicorn process; the unified deployment starts the new code/model.
- GitHub did not serve the frontend. A pre-existing local change mounted the source directory at `/ui`. That change was preserved in a Git stash and a patch before the teammate commit was pulled. The resulting mount now serves only `evaluator_ui/dist`, with no dataset/model/reference directories.
- The dashboard previously required a separate API URL. FastAPI now supplies `/ui/runtime-config.js`, selecting the current browser origin automatically. The standalone static runtime config retains manual URL behavior. Hosted mode ignores stale saved localhost addresses. Frontend and API share HTTPS through ngrok, avoiding mixed content and cross-origin configuration for the unified URL.
- Browser requests to `/` redirect to `/ui/`; non-browser JSON health callers retain the legacy root health response. Versioned health remains `/api/v1/health/live` and `/ready`.
- Leaflet is bundled locally in the teammate integration; OpenStreetMap background tiles still require network access.
- Top-1 and candidate-miss metrics use accepted estimated reference alignments. They are not independent per-observation ground truth. This qualifier is now visible in completed-result warnings and metric subtext. Hidden reference IDs/geometry remain server-side.
- The teammate's fixed 99.8% candidate parity claim was removed from runtime warnings because it is environment-dependent and conflicts with the recorded Mac parity report. The existing user-authorized demo policy is retained and its limitation is explicit.
- JSON preprocessing now rejects infinite scales as well as nonpositive scales. The model uses its exported preprocessing parameters, not a refitted or substituted scaler.
- Mac deployment required the LightGBM 4.7.0 package and OpenMP runtime. No training is part of startup.

## Run and share

From the repository root:

```bash
npm --prefix evaluator_ui run build
.venv/bin/python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal, unless ngrok already forwards to port 8000:

```bash
ngrok http 8000
```

Share the ngrok HTTPS URL with `/ui/` appended. The other computer needs only a browser. The backend/model/data stay on this laptop; no local installation or separate frontend server is needed on the other computer. If ngrok displays its free-tier visitor notice, the viewer may need to continue through that notice once.

Keep the backend process, ngrok process, computer and network connection running. A sleeping/offline laptop stops the service. In-memory jobs are cleared by backend restart. This is a public demo URL with no application login; distribute it only to intended demo viewers and stop the tunnel when finished. It is not a hardened production deployment. Existing explicit CORS settings remain for standalone LAN mode; unified mode uses the same origin and needs no wildcard.

## Review scope

The user previously requested no further test-suite runs. This request was handled through source review, the required frontend build, model startup, and live public HTTP/inference checks. Browser control was unavailable in this chat, so the final page was not visually inspected. Existing test suites were not run. Historical accuracy/parity claims from teammate reports are not presented as independently revalidated results.

## Observed live deployment

The existing ngrok agent was reused and continues forwarding to port 8000. The replacement service binds 127.0.0.1:8000, runs one Uvicorn worker and loads lightgbm-600trees-seed42 with model_ready=true. Public HTTPS checks returned 200 for the dashboard, dynamic same-origin config, app module, bundled Leaflet, health and trip list (10 trips). One genuine HMM reconstruction for reserved trip 00000023 completed through that public API; it was an actual job, not a synthetic fixture. The final page was not visually inspected because browser control was unavailable. No pytest/frontend test suites or retraining were run.

The project virtual environment has LightGBM 4.7.0 installed; Homebrew libomp supplies the Mac OpenMP dependency. The extra temporary startup instance on port 8011 is stopped; the unified server and pre-existing ngrok agent remain running for the demo. The preserved pre-update UI-mount change remains recoverable from the Git stash and `/private/tmp/pathsnap-pre-unified-main.patch`.

### Observed reconstruction quality (one trip only)

Trip 00000023: 458 observations; HMM estimated-alignment Top-1 67.03%, classifier-only Top-1 70.96%, directed continuity 94.97%, unique reference-arc recall 89.83%, candidate miss rate 25.33%, measured inference stage 206.10 ms. The HMM improved neither proven point accuracy nor universal route quality here: its estimated-label Top-1 was lower than the classifier on this trip. These are runtime observations on one reserved trip, not a full-dataset benchmark or a verified-ground-truth accuracy claim. No hidden reference arc IDs or route geometries were exposed.
