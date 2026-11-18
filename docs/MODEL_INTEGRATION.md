# LightGBM handoff

No LightGBM model exists here. Runtime code supports loading only a **supplied trained** native Booster or joblib Booster/LGBMClassifier. Existing historical joblib artifacts are not loaded. Tests use explicit synthetic loader doubles, not real LightGBM training. Genuine library/artifact compatibility and model quality remain untested until delivery.

## Model contract

Use `models/model_manifest.example.json` as the strict template. Place an edited copy at `models/model_manifest.json` or set `PATHSNAP_MODEL_MANIFEST` to an absolute path. Relative artifact/preprocessing paths resolve relative to the manifest directory. Both local model and manifest are ignored by Git.

Required fields: configured=true, name=lightgbm, version, artifact_format (`lightgbm_native` or `joblib_lightgbm`), artifact_path, SHA-256, feature_schema_version=notebook-features-v2, runtime_pipeline=authoritative-notebook-v2, exact ordered feature_names, exact feature_dtypes, training_configuration_reference, output_type=binary_probability, positive_class=1, missing_value_policy. Creation metadata should include training/export versions and date. Calibration metadata records method and permitted development validation; no calibration is assumed.

Use the 15-feature allowlist/order in `backend/app/models/contract.py` and `DATA_CONTRACT.md`. Construct training DataFrames with those names; a model exported from an unnamed array with `Column_0` feature names is incompatible. `label`, identifiers, timestamp, alignment diagnostics and correctness indicators must not enter the predictive matrix.

Native export uses an already-trained Booster's `save_model`; sklearn LightGBM export uses trusted `joblib.dump` on the already-fitted classifier. The native loader calls `lightgbm.Booster(model_file=...)` and `predict(raw_score=False, validate_features=True)`. The sklearn wrapper uses `predict_proba` and class-1 probabilities, checking classes [0,1], output shape N×2 and row probability sums. A joblib Booster uses the native prediction API. Both branches require at least one trained tree, a binary logistic objective, exact stored feature names/count/order, manifest hashes and finite [0,1] output. Multiclass, ranking, raw-margin, arbitrary sklearn pipelines and non-LightGBM artifacts are rejected. See official [Booster API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html) and [LGBMClassifier API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html).

Only load trusted locally supplied pickle/joblib files; checksums detect mismatch, not malicious provenance. Record/pin the training and runtime library versions. Install `requirements-model.txt` only when needed. Missing/invalid artifacts leave the application usable for preview with model state UNCONFIGURED/INVALID and ready=false. Startup validation logs details to the backend; public metadata omits artifact paths and tracebacks. There is no public model upload or reload endpoint. Restart to reload.

## Missing values and preprocessing

Raw LightGBM with native missing handling is supported under `native_missing`; only the four documented nullable columns may contain NaN. `reject` rejects every NaN. Infinity is always forbidden.

For `preprocessor`, provide preprocessing_path, preprocessing_sha256, preprocessing_feature_names and preprocessing_training_reference. The fitted transform must preserve a finite N×15 shape and never fit/refit at inference. The supplied scaler uses median imputation followed by standardization, saved in sklearn 1.6.1. Its order differs at positions 4 and 5:

```text
dist_to_edge_m, dist_to_u_node_m, dist_to_v_node_m,
bearing_diff_deg, bearing_valid, time_gap_prev_s, speed_implied_mps,
irregular_gap_flag, candidate_rank, dist_minus_best_at_point_m,
num_candidates_at_point, edge_length_m, u_node_degree, v_node_degree,
reverse_edge_exists
```

The adapter transforms in that preprocessing order, names its output accordingly and restores canonical model order (bearing_diff_deg before bearing_valid). Training must have used the same transformation and final model order. If your training code uses another final order, fix the export contract before loading; do not hide the mismatch. The supplied medians/means match all usable training rows exactly, but CV fold fitting provenance is not supplied. Fit fold preprocessing on each fold's training partition during development, separately from final export.

## Evaluator features and demo policy

The supplied `PathSnap_Dataset_Pipeline_Final_Pointwise.ipynb` now provides the original generator source. Reviewed definitions are extracted unchanged into `backend/app/pipeline/notebook_features.py`; bootstrap, extraction, fitting and dataset write cells are never executed by the backend. Generated candidates use only raw track/nodes/arcs and flow through the same strict schema/group/mapping validation as precomputed rows.

Full saved-dataset parity does not pass in this Mac runtime: geometric distance values agree within tolerance, but candidates at equal-distance cutoffs can differ. The user authorized demo use despite that limitation. `PATHSNAP_NOTEBOOK_DEMO_MODE=1` is the default, with explicit warnings in health/configuration/result surfaces. Set it to 0 to require a complete passing runtime-bound parity report before generation. Do not claim that demo mode fixes the mismatch, or that a trained model’s quality has been verified with these runtime candidate choices.

Precomputed train/student_test/inference Parquet sources remain supported. Each selected trip must have raw track/nodes/arcs SHA-256 bindings in `trip_bindings.json`. The inference projection reads only metadata and allowlisted features, never labels/diagnostics. Example:

```json
{"00000023":{"track":"SHA256_TRACK","nodes":"SHA256_NODES","arcs":"SHA256_ARCS"}}
```

The model order is now the notebook's explicit FEATURE_COLUMNS, including bearing_diff_deg before bearing_valid. The provisional v1 contract used physical Parquet order; v1 manifests must be updated and any actual model must have stored names matching v2. No real trained artifact existed during that change.

See [NOTEBOOK_INTEGRATION.md](NOTEBOOK_INTEGRATION.md) for the evidence, limitations and source mapping.

## Scoring and decoding

Each output probability stays associated with its original row, point_idx and arc ID. Groups are ranked independently; ties preserve source row order. Binary candidate probabilities are independent and do not sum to one within a GPS group. Decoder emission potential is `log(max(p, 1e-12))`. Normalizing each group would add a state-independent constant and leave Viterbi argmax unchanged, so no extra normalization is applied. These are discriminative potentials, not a claim of calibrated generative observation probabilities. Any calibration, spatial reweighting or transition tuning needs permitted development evaluation after training.

The historical decoder is preserved for offline experiments, but production decoding receives only grouped scores, directed graph and timestamps. Allowed transitions are identical arcs or `v(previous)==u(current)` at equal log cost 0; forbidden transitions cost negative infinity. No arbitrary finite disconnected penalty is used. This conservative decoder does not infer shortest-path intermediate edges between sparse observations. Runs reset at time gaps >30 seconds (an operational policy, not learned), nonincreasing timestamps, or no feasible directed transition; diagnostics mark every break. Connectivity is guaranteed only within each decoded run. Continuity is the fraction of all consecutive predicted observations on identical/adjacent directed arcs; gaps still count in the denominator. Same-arc repeats do not certify forward movement within an arc because no authoritative along-arc projection feature is available. All repeated predictions and segment order are preserved; no bridges are drawn between segments.

## Validation sequence

```bash
python -m pip install -r requirements-model.txt
python scripts/validate_model.py
python scripts/validate_model.py --trip 00000023
python -m pytest -q
# Restart single-worker API, then check /api/v1/health/ready and trip metadata.
```

The loader probes outputs on five **actual** rows projected from the supplied student-test/train data; it does not invent production probe features. The CLI exits nonzero until model readiness is true. Full trip validation also checks checksum bindings and candidate grouping. After delivery, run an actual reconstruction and inspect backend stage logs, actual arc sequences, directed breaks, latency and aggregate evaluation. Compare classifier-only versus HMM on permitted development trips before making quality claims. The supplied reference aligner has a remaining parity difference: top1_accuracy and candidate_miss_rate currently remain null. Set Jaccard against the protected reference arc sequence is available after genuine inference; it is not sequence edit accuracy.
