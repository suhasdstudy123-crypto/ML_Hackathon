# Supplied notebook audit — 18 November 2026

Source: `PathSnap_Dataset_Pipeline_Final_Pointwise.ipynb`, supplied by the user after the first implementation. SHA-256: `9fa739dc905d93e7d46fa31eeb1fad9d4d4916c337ce0bdfec804dec766b1b5d`. All 26 cells were read; all execution counts are null and the notebook contains no saved outputs. Its markdown is source documentation, not an instruction to execute installation, extraction, fitting, or dataset writes.

## Decisions before integration

| Cells / component | Decision | Findings and integration action |
|---|---|---|
| Original notebook | KEEP | Preserve byte-identical source under notebooks, with source hash and extraction provenance. Do not execute the full notebook. |
| 2–5 environment/bootstrap | ARCHIVE in source only | Contains pip installs, automatic ZIP extraction and directory discovery/creation. Do not import these into backend. Read only literal constants needed by pure algorithms. |
| 7 geographic helpers | ADAPT without algorithm edits | Equirectangular node-centroid projection (R=6,371,000m), great-circle bearings, haversine speeds. Extract reviewed definitions only. |
| 9 raw loader | KEEP in notebook | Reads `.route`; unsuitable for inference. Retain existing validated TripService loading only track/nodes/arcs. Adapt arrays into a route-blind dataclass. |
| 11 candidate generation | ADAPT without algorithm edits | KD-tree midpoint pool of max(4k,20), capped at graph size; rerank by segment distance; k=5. Contrary to broad markdown wording, finite overfetch cannot guarantee globally nearest segments. Preserve candidate/tie behavior and verify against Parquet. |
| 13 feature engineering; 18 build_trip_table | ADAPT without algorithm edits | All 15 predictors are route-blind. Canonical model order is explicit FEATURE_COLUMNS, with bearing_diff_deg before bearing_valid. Physical Parquet column order differs. Version the prior provisional model contract rather than silently changing its meaning. |
| 14 monotonic reference alignment | ADAPT for evaluation only | Route-position DP, repeated arc positions, Huber geometry/heading costs and uncertainty flags. It estimates observation labels, not independent pointwise ground truth. Keep in separate evaluation module, read `.route` only after inference, return aggregates only, bound O(T×R) allocation. |
| 16 labels and QA | KEEP in notebook / reuse inspected semantics | Accepted alignment AND candidate hit produces supervised groups; all other labels remain NaN. Estimated candidate miss rate denominator is accepted alignments. Never use candidate membership to guide alignment. |
| 20 CV/preprocessing; 22 full orchestration | KEEP, do not execute | Source fits median-impute/scaler on non-null training-label rows. GroupKFold is group-aware, but final global scaler does not itself guarantee fold-only preprocessing. No training, fitting, refitting, or authoritative artifact writing is authorized here. |
| 24–25 inspection/download | KEEP in notebook only | No production imports. |
| Raw Student folder | READ for parity only | Found 80 train and 10 student-test directories locally corresponding to supplied Parquet trip IDs. Do not copy into Git or alter raw files. Verify candidate IDs/order, values/dtypes/null masks, alignments and labels before claiming reproduction. |

## Integration gates

1. Preserve all prior dataset/model checksums and original notebook bytes.
2. Extract reviewed pure definitions into source modules; regression tests compare function ASTs to notebook cells so algorithm changes cannot hide in the adapter.
3. Verify parity against supplied feature/alignment artifacts using corresponding raw files. Any mismatch is recorded explicitly; never replace the source datasets to obtain a passing result.
4. Generate evaluator candidates only from raw track/nodes/arcs. No protected route in the inference input object. Model readiness remains false until a genuine trained artifact is supplied.
5. If alignment parity passes, report accuracy as an **estimated aligned-label metric**, including denominators and uncertainty coverage. Never claim verified independent pointwise accuracy.
