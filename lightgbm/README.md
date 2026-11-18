# PathSnap — LightGBM candidate classifier (baseline)

Candidate-scoring model for PathSnap map-matching: for every (GPS point, candidate road arc) pair it predicts the probability that the arc is the one the vehicle was on.

## Results (seed 42)
| | Macro F1 | Top-1 accuracy | Log loss |
|---|---:|---:|---:|
| 5-fold GroupKFold CV (80 training trips) | 0.8643 ± 0.0233 | 0.7434 ± 0.0405 | 0.1947 |
| Student-testing trips (10, held out) | 0.8814 | 0.7719 (0.8170 on covered points) | 0.1708 |

Route continuity: not computed (raw road graphs not attached in this run).

## Contents
| Path | What it is |
|---|---|
| `notebooks/pathsnap_lightgbm.ipynb` | Code that produced everything here (setup, labels, metrics, LightGBM) |
| `models/lightgbm.joblib` | Full sklearn pipeline: median imputer → standard scaler → LGBMClassifier |
| `models/lightgbm_booster.txt` | Raw LightGBM booster (version-independent; apply imputer+scaler yourself) |
| `models/lightgbm_model_card.json` | Feature order, preprocessing, seed, loading instructions |
| `results/lightgbm_results.json` | All metrics, hyperparameters, per-fold CV, confusion matrix |
| `results/lightgbm_cv_folds.csv` | Per-fold CV metrics and timings |
| `results/lightgbm_confusion_matrix.png` | Student-test confusion matrix |
| `results/lightgbm_student_test_proba.parquet` | Student-test probability per candidate |
| `results/lightgbm_oof_train.parquet` | Out-of-fold training probabilities (for HMM tuning without leakage) |
| `PathSnap_Model_Report.md` | Comparison report (LR / RF / LightGBM) |

## Hyperparameters
600 trees, learning_rate 0.05, num_leaves 63, min_child_samples 50, subsample 0.8 (freq 1), colsample_bytree 0.8, reg_lambda 1.0, objective binary.

## Load
```python
import joblib
model = joblib.load("models/lightgbm.joblib")          # needs the versions in requirements.txt
proba = model.predict_proba(X[FEATURE_COLUMNS])[:, 1]  # FEATURE_COLUMNS: see lightgbm_model_card.json
```

## Reproduce
Inputs: `train_features.parquet`, `student_test_features.parquet`, `train_alignment_diagnostics.parquet`, `student_test_alignment_diagnostics.parquet` (output of the parsing pipeline). Run `notebooks/pathsnap_lightgbm.ipynb` with these under `/kaggle/input` (or set `PATHSNAP_INPUT`).
