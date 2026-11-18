# Notebook integration and demo policy — 18 November 2026

The user supplied `PathSnap_Dataset_Pipeline_Final_Pointwise.ipynb` and subsequently instructed: proceed for the demo despite unavailable original runtime details, stop further testing, and push the implementation. The final demo-mode/configuration changes were therefore **not tested**.

## Implemented

- Original notebook retained byte-identical under `notebooks/`. SHA-256: `9fa739dc905d93e7d46fa31eeb1fad9d4d4916c337ce0bdfec804dec766b1b5d`.
- Reviewed cells 7, 11, 13 and the route-blind function in cell 18 extracted without algorithm changes into `backend/app/pipeline/notebook_features.py`. The extraction script parses source; it never executes installation, ZIP extraction, raw-data discovery, preprocessing fit or full-build cells.
- Cell 14's reference aligner retained separately in `reference_alignment.py`, excluded from inference. The production generator’s TripData contains only trip_id/nodes/arcs/track, no route field.
- FeatureService retains precomputed feature support with raw-file bindings. For trips without precomputed rows, it now invokes the supplied notebook generator according to the demo/strict policy and validates feature names/order/types, missing values, five candidates per observation, candidate ranks, timestamps and trip-local arc mappings.
- GENERATING_CANDIDATES is a real job stage only for online generation. Generated-feature warnings propagate to completed results. Raw data parsing uses round-trip float conversion to avoid avoidable parser precision loss.
- Model contract corrected to the notebook's explicit order: bearing_diff_deg precedes bearing_valid. Schema is **notebook-features-v2**, runtime **authoritative-notebook-v2**. The physical Parquet column order is unchanged. The old provisional v1 manifest is rejected rather than silently reinterpreted. No trained model existed when this correction was made.

## Demo configuration

`PATHSNAP_NOTEBOOK_DEMO_MODE=1` is the user-authorized default. It permits the unchanged reviewed notebook generator despite the known saved-dataset parity differences. The source hash remains checked; demo mode does not allow an arbitrary substitute pipeline. Health reports the demo policy; selected-trip feature availability and generated-feature results carry the limitation. Candidate geometry, scores and metrics are never fabricated.

Set `PATHSNAP_NOTEBOOK_DEMO_MODE=0` for strict mode. Strict mode requires a complete passing report under `PATHSNAP_PARITY_REPORT` (default docs/NOTEBOOK_PARITY.json), covering the exact 80 training and 10 student-test trip IDs, source artifact hashes, notebook/module hashes and the current runtime. The existing report records failure, so strict online generation remains unavailable. Precomputed exact trip features with valid bindings remain usable independently.

No LightGBM training, scaler fitting/refitting or source dataset writes occurred. Reconstruction remains MODEL_NOT_READY until a genuine compatible trained artifact and v2 manifest are supplied. Top1 accuracy/candidate miss evaluation remain null while the alignment reproducibility limitation is unresolved; aggregate unique-arc Jaccard is supported after real inference.

## Evidence collected before the stop-testing instruction

All 90 train/student-test trips were compared using existing raw files under the locally available Student folder. The authoritative artifacts were read, not regenerated or replaced. `NOTEBOOK_PARITY.json` records the full comparison, raw-input checksums, runtime and module provenance.

| Comparison | Train | Student test |
|---|---:|---:|
| Observations | 208,417 | 19,160 |
| Candidate rows | 1,042,085 | 95,800 |
| Candidate identity row mismatches | 409,803 | 37,169 |
| Observations with candidate-set differences | 86,792 | 7,930 |
| Estimated binary label row mismatches | 123,346 | 13,655 |

Geometric distance columns match within rtol=1e-12 / atol=1e-8, while candidate choices at ties, ranks and dependent features differ. One reference-aligned position/arc differs (train trip 00000011). Example: trip 00000001 observation 0 has tied candidates 50652/53932 reordered, and its fifth candidate differs between 50653 and 53933 at the same distance. The raw graph also contains duplicate directed endpoint pairs. These are genuine identity changes, not just a shuffled display table.

Cause is not established. Unstable tie selection, numerical execution differences and original export environment are possible contributors. NumPy 1.26.4 versus 2.2.5 on available ARM environments did not resolve the first checked trips. Changing sort policy would alter the original generator, so it was not done. An optional isolated x86 dependency installation was not executed because automatic approval review failed with a usage-limit error; this is an environment limitation, not evidence that the install is unsafe. The user then authorized moving on with demo mode.

Before the final demo-mode changes: 70 backend tests passed; 15 frontend tests, type checks and the static build passed. Those results do not validate the subsequent untested demo-policy, float-parser and documentation edits. Testing stopped on explicit user instruction. Previous browser/LAN verification is documented in DEPLOYMENT.md; no new browser run was performed for the final edits.

## Model connection

Use `models/model_manifest.example.json` as the v2 template. Export an already-trained LightGBM model with the exact named FEATURE_COLUMNS order, record artifact hashes/version/training reference and select the preprocessing policy used during training. If the supplied scaler is expected, its input order now matches the notebook/model order. Set configured=true, install the optional runtime dependencies when needed, and restart the backend. Validation/real-inference checks should be run later when the user authorizes testing and the actual model is available. No frontend rewrite is required.
