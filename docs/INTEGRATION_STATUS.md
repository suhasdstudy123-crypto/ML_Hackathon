# Integration status — 18 November 2026

The FastAPI/Leaflet infrastructure and supplied notebook adapter are implemented. Genuine inference remains pending the user's trained LightGBM artifact. **The final demo-mode changes were not tested, per the user's explicit instruction to stop testing and push.**

## Current behavior

- All ten real evaluator trips remain available for raw GPS and directed-road previews.
- The supplied notebook is preserved and its reviewed route-blind generator is integrated. Missing evaluator precomputed features no longer require inventing another pipeline: demo mode generates them using the original source.
- `PATHSNAP_NOTEBOOK_DEMO_MODE=1` defaults to the user-authorized demo policy. It accepts documented candidate-tie differences from the original export and carries a warning. Set 0 for strict complete-parity enforcement. Strict mode currently fails the saved parity report.
- Model schema/order is corrected and versioned `notebook-features-v2`, runtime `authoritative-notebook-v2`: bearing_diff_deg before bearing_valid, matching notebook FEATURE_COLUMNS and its scaler.
- No model is trained, substituted or simulated. model_ready remains false, and reconstruction returns 503 MODEL_NOT_READY without a genuine trained artifact.
- Inference receives no `.route`/reference data. Alignment source is isolated; top1/miss metrics remain null pending accepted alignment reproducibility. Evaluation can report aggregate unique-arc Jaccard only after genuine inference.

## Preserved and changed

All original datasets, raw evaluator files, historical feature engineering, training/tuning/EDA utilities, saved models, NPZ caches, diagnostic/results tables and map/plot artifacts were retained. The seven supplied ZIP artifacts and original notebook were not modified. Historical README is archived under docs/history. No data/model artifacts were deleted or regenerated.

Added modular backend services, model manifest/provider, candidate validation, directed decoder, bounded jobs, typed frontend API client, responsive OSM dashboard, configuration/import/validation tools, notebook extraction/provenance and deployment documentation. Production no longer imports old classifiers or exposes hidden reference arcs/geometries. Old Windows paths, import-time model loading, unsafe wildcard credentialed CORS and browser fetch receiver were fixed in earlier commits.

## Reproduction findings

The full 80+10-trip comparison completed before the user stopped testing. Geometric distances agree within tolerance; candidate identity differs on 446,972 rows, and 94,722 observations have a changed candidate set. One aligned position/arc also differs. No sort or feature algorithm was changed to hide this. The user elected demo mode with this known limitation. Details: [notebook integration](NOTEBOOK_INTEGRATION.md), [audit](NOTEBOOK_AUDIT.md), [machine-readable parity evidence](NOTEBOOK_PARITY.json).

## Checks actually run before testing stopped

- Initial implementation: 57 backend tests passed; 15 frontend tests passed; typecheck/build and live frontend-client/API check passed.
- Notebook/strict-gate checkpoint before final demo edits: **70 backend tests passed in 3.95 seconds**; 15 frontend tests, TypeScript checks and static build passed.
- Prior artifact verification: 69 historical artifacts and seven authoritative inputs matched original SHA-256 checksums.
- Previous browser check verified actual GPS/road previews, trip selection, layer controls, disabled model-pending state, reconnect and same-machine loopback/LAN-address communication. Physical second-laptop testing was not available.
- Full notebook parity comparison: **failed**, as documented; it is not counted as a passing integration check.

No further tests, builds, browser checks or artifact validation runs were conducted after the stop-testing instruction. Final demo policy, raw-float parser and documentation edits are unverified. The three preserved legacy exploratory scripts still require their old Windows training paths. No genuine LightGBM artifact/library compatibility, prediction quality, calibration or full inference performance check has run because no model was supplied. An optional x86 environment install was blocked by automatic approval review's usage limit; work continued without it.

## Handoff

1. Supply the already-trained LightGBM native/joblib artifact.
2. Use the updated v2 manifest example; set real format/path/hash/version, exact notebook feature names/dtypes, training reference, missing-value policy and optional fitted preprocessing provenance. Keep configured=false until real export is ready.
3. Run the backend using the documented single-worker LAN command and explicit frontend origin. Leave demo mode 1 for the requested demo, or use strict mode only after reproducibility is resolved.
4. When testing is authorized again, run compatibility validation and actual prediction/HMM/evaluation checks with that genuine artifact. Perform physical two-laptop connectivity testing before demonstration.

See [README](../README.md), [deployment](DEPLOYMENT.md), [model integration](MODEL_INTEGRATION.md), [data contract](DATA_CONTRACT.md) and [API contract](API_CONTRACT.md).
