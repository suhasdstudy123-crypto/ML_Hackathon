# Authoritative data contract

Inspected 18 November 2026 from user-provided `outputs (1).zip`. The seven original artifacts were copied byte-for-byte into ignored `data/authoritative/`. `AUTHORITATIVE_CHECKSUMS.json` records their SHA-256 values; `DATA_SCHEMA_INSPECTION.json` records every column, dtype and null/infinity count. The ZIP and historical caches remain unchanged.

## Ordered predictive features — authoritative-parquet-v1

This is the inference allowlist, in original Parquet order after removing metadata and label. Final training must explicitly adopt this order, or the model will be rejected. This is an integration contract, not a claim that an unsupplied training script already uses it.

| Order | Name | dtype | Meaning / inference availability |
|---|---|---|---|
| 1 | dist_to_edge_m | float64 | GPS-to-candidate segment distance, metres |
| 2 | dist_to_u_node_m | float64 | GPS-to-directed start node distance, metres |
| 3 | dist_to_v_node_m | float64 | GPS-to-directed end node distance, metres |
| 4 | bearing_valid | int64 | Bearing validity indicator |
| 5 | bearing_diff_deg | float64 | Absolute bearing difference, degrees; nullable |
| 6 | time_gap_prev_s | float64 | Previous GPS time gap, seconds; nullable at first observation |
| 7 | speed_implied_mps | float64 | GPS implied speed, metres/second; nullable |
| 8 | irregular_gap_flag | float64 | Irregular time interval flag; nullable |
| 9 | candidate_rank | int64 | Zero-based candidate rank 0–4 |
| 10 | dist_minus_best_at_point_m | float64 | Distance above closest candidate, metres |
| 11 | num_candidates_at_point | int64 | Candidate count, exactly 5 in supplied data |
| 12 | edge_length_m | float64 | Directed segment length, metres |
| 13 | u_node_degree | int64 | Start-node degree; exact directed/undirected convention needs generator |
| 14 | v_node_degree | int64 | End-node degree; convention needs generator |
| 15 | reverse_edge_exists | int64 | Whether reverse directed edge exists |

All fields are numerical. Integer indicators remain numerical (no categorical coding inferred). Data are unscaled physical values as shown by coordinate-distance magnitudes. Exact distance projection, tie breaking, heading behavior, graph degree semantics, irregular-gap threshold and candidate generation cannot be recovered reliably from column names. **The original generator is missing. Do not recreate it by guessing.**

The Parquet column order is: `trip_id`, `point_idx`, `candidate_edge_id`, `timestamp`, the 15 features above, `label` (20 columns total). See machine-readable inspection for every dtype. `trip_id` is string/object and must retain leading zeros; `point_idx` and `candidate_edge_id` are int64; `timestamp` is float64. Timestamp is available at inference but deliberately used only as observation metadata. Labels and all alignment diagnostic columns are forbidden in model inputs.

## Targets, grouping and alignment

Groups are `(trip_id, point_idx)`, candidate identity is `(trip_id, point_idx, candidate_edge_id)`. Point indices are chronological track row positions. Each candidate ID is a row index into that trip's directed `.arcs`; endpoints are row indices into `.nodes`. Never reuse arc IDs across trips. Inference requires raw-file SHA-256 bindings in `trip_bindings.json`, matching timestamps, valid node/arc indices, unique candidate mappings, contiguous covered observations, five candidates per group and ranks 0–4. Candidate rows stay associated with IDs throughout scoring and ranking.

`label`: float64, 1 for aligned positive edge and 0 for negative candidates, **only for usable aligned groups**. Other groups have all five labels NaN; these must not become all-negative training examples. Joining diagnostics confirms usable groups have exactly one positive candidate. No reference-derived field is a predictor.

| Inspection | Train | Student test |
|---|---:|---:|
| Trips | 80 | 10 |
| Observations | 208,417 | 19,160 |
| Candidate rows | 1,042,085 | 95,800 |
| Positive labels | 175,984 | 18,079 |
| Negative labels | 703,936 | 72,316 |
| NaN labels | 162,165 | 5,405 |
| Accepted alignments | 186,690 | 19,136 |
| Uncertain alignments | 21,727 | 24 |
| Diagnostic candidate hits (all alignments) | 178,998 | 18,084 |
| Usable labeled groups | 175,984 | 18,079 |

Diagnostic columns, in order: `trip_id` object, `point_idx` int32, `aligned_route_pos` int32, `aligned_edge_id` int64, `alignment_distance_m` float64, `aligned_route_distance_m` float64, `alignment_status` object, `candidate_hit` bool, `usable_for_training` bool. They are evaluation/training-only, never loaded by inference or served to a browser.

Train statuses: 186,690 accepted, 21,679 uncertain_distance, 37 uncertain_transition, 11 uncertain_route_gap. Test: 19,136 accepted, 21 uncertain_distance, 2 uncertain_transition, 1 uncertain_route_gap. Candidate misses among accepted groups: 10,706 train and 1,057 test (accepted minus usable). CSV miss rates use accepted-alignment denominators, not all observations. Train trips 00000054 and 00000033 have particularly high uncertain alignment rates (87.16% and 51.78%). Final model evaluation must disclose usable-only versus overall coverage denominators.

No infinities were found. Nullable feature counts: train bearing_diff_deg 430, time_gap_prev_s/speed_implied_mps/irregular_gap_flag each 400; test each 50. All other predictive features are non-null. Native LightGBM missing values are permitted only for those four columns under an explicit manifest policy. Unexpected NaN, infinity, dtype changes, missing/extra/reordered columns are rejected.

## Preprocessing

`feature_scaler.joblib` is a fitted sklearn Pipeline: median SimpleImputer then StandardScaler, saved with sklearn 1.6.1. It has 15 inputs and no `feature_names_in_`. Its actual order is **bearing_diff_deg before bearing_valid**, unlike the Parquet order. All other columns retain the order above. Comparing all imputer medians and scaler means to usable labeled training rows in that order produces maximum absolute difference **0.0**, and scaler `n_samples_seen_` is 879,920, exactly five times 175,984. This strongly supports fitting only usable training rows; source provenance and cross-validation fold fitting cannot be proved without the generating script. Existing sklearn 1.7.2 warns about loading the 1.6.1 artifact.

LightGBM does not inherently require scaling. Prefer raw features with native missing-value handling if that is how the final model is trained. If training uses this preprocessing, the manifest must record its hash, explicit input order, training reference and compatible sklearn version. Adapter transforms in preprocessing order, then restores canonical model order. **Never fit/refit at inference.** Training must export model names in canonical order; arrays with unnamed `Column_0` features are rejected.

## Reproduction blocker and supported interface

The historical `feature_engineering.py` produces 10 other features and 10–15 candidates, with a different label aligner. It cannot reproduce this Parquet. It is preserved and excluded from production imports. Only exact precomputed authoritative inference columns are currently supported. `FeatureService` projects metadata and allowlisted features without reading `label` or alignment diagnostics.

None of the ten reserved evaluator trips occurs in the supplied Parquet files. Raw train and student-test graphs are absent. To enable evaluator inference, supply the original generator, or exact evaluator `inference_features.parquet` produced by that generator and raw-file checksum bindings. Do not use `.route` during feature generation. Missing features produce `FEATURE_PIPELINE_UNAVAILABLE`; missing models produce `MODEL_NOT_READY` first.

Geometry uses WGS84 GeoJSON `[longitude, latitude]`; every predicted arc is an individual directed LineString. Repeated predictions are preserved. No synthetic bridge is drawn across disconnected components. Road preview may truncate at 6,000 viewport arcs and reports this explicitly; full model inputs and decoder graph are never downsampled.
