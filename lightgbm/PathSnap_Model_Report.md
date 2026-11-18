# PathSnap — Candidate Classifier Experiments (Baseline Report)

**Project:** PathSnap: Distributed Map-Matching & Route Reconstruction (UE24CS352A)
**Stage:** candidate scoring (before Bayesian optimisation and HMM/Viterbi)
**Run:** Kaggle notebook `pathsnap-all-models`, Version 1 ("5 baseline classifiers"), CPU, seed 42
**Date:** 18 Nov 2026
**Status:** Logistic Regression, Random Forest and LightGBM completed. KNN and MLP were stopped before producing results and still need to be run (the guidelines require ≥ 5 evaluated models).

---

## 1. Task framing

Each GPS observation has up to 5 candidate road arcs (`K_CANDIDATES = 5`). Every (observation, candidate) pair is one row, and a binary classifier scores how likely that candidate is the arc the vehicle was actually on. The highest-scoring candidate per observation is the classifier-only prediction. Later, these probabilities become the HMM emission scores.

## 2. Data used

| Partition | Trips | Observations | Candidate rows | Labelled rows | Candidate miss rate | Uncertain alignments |
|---|---:|---:|---:|---:|---:|---:|
| Training (fit + 5-fold CV) | 80 | 208,417 | 1,042,085 | 84.4% | 14.12% | 10.42% |
| Student-testing (held out) | 10 | 19,160 | 95,800 | 94.4% | 5.62% | 0.13% |

The 10 evaluator trips were not used anywhere.

### Labels
- The `label` column from the parsing pipeline is point-level: each GPS point is aligned to the ordered `.route`, and the candidate equal to the aligned arc gets `1`, all others `0`.
- Points whose reference arc is not among the candidates, or whose alignment is uncertain, have `label = NaN`. They are excluded from fitting and from the classification metrics.
- Sanity checks passed on both partitions: label agrees with the reference arc on 100% of labelled rows, and every labelled point has exactly one positive.
- Labelled rows are about 20% positive (1 of 5 candidates).

### Features (15)
| Group | Features |
|---|---|
| Spatial | `dist_to_edge_m`, `dist_to_u_node_m`, `dist_to_v_node_m` |
| Directional | `bearing_diff_deg`, `bearing_valid` |
| Temporal | `time_gap_prev_s`, `speed_implied_mps`, `irregular_gap_flag` |
| Candidate-relative | `candidate_rank`, `dist_minus_best_at_point_m`, `num_candidates_at_point` |
| Road topology | `edge_length_m`, `u_node_degree`, `v_node_degree`, `reverse_edge_exists` |

Metadata (`trip_id`, `point_idx`, `timestamp`, `candidate_edge_id`) is not used as a feature.

### Preprocessing (identical for every model)
Median imputation followed by standard scaling, inside a scikit-learn `Pipeline`, so it is fitted on the training folds only within each CV split. Scaling has no effect on tree models but keeps the pipeline the same for every model.

## 3. Experimental protocol

1. **Cross-validation:** `GroupKFold(5)` grouped by `trip_id` on the 80 training trips, so no trip is split between train and validation.
2. **Refit:** the final model is fitted on all labelled rows of the 80 training trips.
3. **Held-out evaluation:** the refitted model is scored on the 10 student-testing trips.
4. **Over/underfitting check:** the refitted model is also scored on 200,000 labelled training rows (in-sample).
5. **No class weighting**, so predicted probabilities stay usable as HMM emissions.
6. **Logged:** fit time per fold, final training time, and inference time per 1,000 rows.

### Metrics
| Metric | Definition |
|---|---|
| Accuracy / Precision / Recall / F1 / Macro F1 | Row-level, threshold 0.5, labelled rows only |
| Log loss | Row-level probability quality, labelled rows only |
| **Observation-level Top-1 accuracy** | For each point with an accepted alignment, the highest-scoring candidate must equal the reference arc. Candidate misses count as wrong. |
| Top-1 (covered) | Top-1 restricted to points whose reference arc is in the candidate set |
| Candidate miss rate | Share of points whose reference arc is not among the candidates |
| Route continuity | Share of consecutive predictions on the same or directly connected arcs. **Not computed in this run** because the raw `.arcs` files were not attached (reported as N/A). |

**Primary selection metric:** observation-level Top-1 accuracy, because it measures what the classifier must actually do: choose the right arc for each GPS point. **Log loss is the secondary metric**, because the probabilities feed the HMM.

## 4. Consolidated comparison

| Model | Algorithm family | Key configuration | CV Macro F1 | CV Top-1 | CV Log loss | Student-test Top-1 | Route continuity | Student-test Macro F1 | Student-test Log loss | Train time | Inference |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| Logistic Regression | Linear / probabilistic | L2, C=1.0, lbfgs, max_iter=2000 | 0.852 ± 0.024 | 0.736 ± 0.042 | 0.229 | 0.763 | N/A (no road graph attached) | 0.872 | 0.210 | 11.1 s | 0.24 ms / 1k rows |
| Random Forest | Ensemble (bagging) | 300 trees, max_depth=20, min_samples_leaf=10, max_features=sqrt | 0.863 ± 0.024 | 0.740 ± 0.040 | 0.198 | 0.765 | N/A | 0.880 | 0.176 | 339.2 s | 15.96 ms / 1k rows |
| **LightGBM** | Boosting | 600 trees, lr=0.05, num_leaves=63, min_child_samples=50, subsample/colsample 0.8, λ=1.0 | **0.864 ± 0.023** | **0.743 ± 0.041** | **0.195** | **0.772** | N/A | **0.881** | **0.171** | 25.0 s | 16.14 ms / 1k rows |
| Distance-weighted KNN | Instance-based | k=25, weights=distance, KD-tree, ≤ 300k training rows | not completed | — | — | — | — | — | — | — | — |
| MLP | Neural network | (64, 32) ReLU, Adam lr=1e-3, α=1e-4, batch 1024, early stopping | not completed | — | — | — | — | — | — | — | — |

Student-test Top-1 on covered points only: LR 0.807, RF 0.810, LightGBM 0.817.

### Per-fold CV results (Top-1 / Macro F1 / Log loss)

| Fold | Logistic Regression | Random Forest | LightGBM |
|---|---|---|---|
| 0 | 0.688 / 0.829 / 0.265 | 0.692 / 0.844 / 0.229 | 0.695 / 0.846 / 0.224 |
| 1 | 0.705 / 0.837 / 0.243 | 0.712 / 0.847 / 0.214 | 0.715 / 0.848 / 0.214 |
| 2 | 0.731 / 0.839 / 0.229 | 0.738 / 0.849 / 0.204 | 0.741 / 0.851 / 0.204 |
| 3 | 0.786 / 0.886 / 0.194 | 0.787 / 0.897 / 0.161 | 0.793 / 0.899 / 0.153 |
| 4 | 0.770 / 0.869 / 0.215 | 0.773 / 0.877 / 0.181 | 0.774 / 0.877 / 0.178 |
| Fit time per fold | 5.7–8.6 s | 245.6–283.4 s | 19.7–22.0 s |

The ranking LightGBM ≥ Random Forest > Logistic Regression holds on every fold. Folds differ much more from each other (about 10 Top-1 points between fold 0 and fold 3) than the models differ, so trip difficulty dominates the variance.

### Train vs. validation (overfitting evidence)

| Model | Train Macro F1 | CV Macro F1 | Gap | Train Log loss | CV Log loss |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.854 | 0.852 | 0.002 | 0.224 | 0.229 |
| Random Forest | 0.893 | 0.863 | 0.030 | 0.154 | 0.198 |
| LightGBM | 0.895 | 0.864 | 0.031 | 0.153 | 0.195 |

## 5. Per-model analysis

### 5.1 Logistic Regression
- **Why it suits the problem:** a fast, well-calibrated linear baseline. It shows how much of the task is solved by a weighted sum of distance, bearing and rank features.
- **Strengths / limitations:** very fast (11 s to train, 0.24 ms per 1k rows) and interpretable through its coefficients. It cannot model interactions such as "close *and* aligned with the direction of travel" unless they are engineered as features.
- **Hyperparameters:** C = 1.0 (L2), lbfgs, max_iter = 2000, no class weights.
- **Objective:** binary cross-entropy (log loss) with an L2 penalty.
- **Results:** student-test Top-1 0.763, Macro F1 0.872, log loss 0.210. On student-test, 3,697 of 18,079 positive rows fell below the 0.5 threshold. Top-1 is still reasonable because ranking only needs the true arc to score highest.
- **Fit:** almost no train/CV gap (0.002 F1), which points to **underfitting**: the linear decision boundary is the limit.

### 5.2 Random Forest
- **Why it suits the problem:** bagged trees learn non-linear rules and feature interactions, with variance reduced by averaging.
- **Strengths / limitations:** robust and needs little tuning. However it is the slowest to train (339 s; about 250 s per fold) and its saved model is large.
- **Hyperparameters:** 300 trees, max_depth = 20, min_samples_leaf = 10, max_features = sqrt, bootstrap.
- **Objective:** Gini impurity splits; probabilities are averaged leaf class frequencies.
- **Results:** student-test Top-1 0.765, Macro F1 0.880, log loss 0.176. It gains little Top-1 over Logistic Regression (+0.2 points) but clearly improves log loss.
- **Fit:** moderate **overfitting** (train F1 0.893 vs CV 0.863; train log loss 0.154 vs 0.198).

### 5.3 LightGBM
- **Why it suits the problem:** gradient-boosted trees are typically the strongest family on tabular features. LightGBM is fast on about 880k rows and handles missing values natively.
- **Strengths / limitations:** best accuracy and probability quality at about 1/14 of Random Forest's training time. It has more hyperparameters to manage and can overfit without regularisation.
- **Hyperparameters:** 600 trees, learning rate 0.05, num_leaves 63, min_child_samples 50, subsample 0.8 (freq 1), colsample_bytree 0.8, reg_lambda 1.0.
- **Objective:** binary log loss.
- **Results:** student-test Top-1 **0.772**, Macro F1 **0.881**, log loss **0.171**, the best on every metric.
- **Fit:** moderate **overfitting**, similar to Random Forest (train F1 0.895 vs CV 0.864). Regularisation is a natural target for tuning.

### 5.4 Distance-weighted KNN — not completed
- **Planned configuration:** k = 25, inverse-distance weights, Euclidean distance on standardised features, training rows capped at 300k.
- **Search-structure benchmark (completed):** 200k-point index, 5,000 queries.

| Structure | Build | Query (5k points) |
|---|---:|---:|
| KD-tree | 0.63 s | **5.09 s** |
| Ball-tree | 0.45 s | 18.30 s |

KD-tree was chosen, being about 3.6× faster to query. Even so, the expected run time is about 30 minutes, because each fold scores about 200k validation rows at roughly 1,000 queries/s. The run was stopped during KNN's CV.

### 5.5 MLP — not completed
Planned configuration: hidden layers (64, 32) with ReLU, Adam (lr 1e-3), L2 α = 1e-4, batch 1,024, early stopping (10 epochs patience, 10% validation split). The early-stopping split is row-random rather than trip-grouped; this affects only when training stops, not the CV or student-test scores.

## 6. Findings

1. **LightGBM is the best model so far** on the primary metric (student-test Top-1 0.772, CV 0.743), and also on log loss (0.171), which matters for the HMM. It trains in 25 s, which makes Bayesian optimisation with many trials affordable.
2. **The models differ by under 1 Top-1 point**, while folds differ by about 10. Model choice is therefore not the main bottleneck. The current 15 features and the candidate set are.
3. **The candidate set caps achievable Top-1.** 14.1% of training points and 5.6% of student-test points have their reference arc outside the 5 candidates, and no classifier can recover those. Raising `K_CANDIDATES` may help more than switching classifier.
4. **Reverse-direction duplicates:** two-way roads appear as two candidates (u→v and v→u) with identical distance features, so only `bearing_diff_deg` separates them. At the first point of a trip, the bearing is missing and the choice is effectively a coin flip. This is one likely source of the remaining Top-1 errors.
5. **Probability quality improves more than ranking** when moving from linear to tree models: log loss falls about 15–19%, while Top-1 rises only 0.3–0.9 points. This favours tree models for the HMM emission stage.

## 7. Remaining work

| Item | Why |
|---|---|
| Run KNN and MLP | Required: at least 5 models must be trained and evaluated |
| Re-save model artefacts | The cancelled run kept no output files; refit LightGBM (about 25 s) to get the `.joblib` for the API |
| Attach raw trip folders (`pathsnap-raw-trips`) | Enables the route-continuity metric (currently N/A) |
| Bayesian hyperparameter optimisation of LightGBM | Required step; objective = grouped-CV Top-1 (or log loss), compare tuned vs untuned |
| HMM + Viterbi decoding | Required; compare classifier-only vs classifier + HMM on Top-1 and continuity |
| Feature extensions | Heading change, acceleration, projection position along the arc, next-point distance, to separate reverse-direction duplicates and parallel roads |

---
*All numbers are copied from the Kaggle run log of Version 1 (cancelled after LightGBM completed). Training-set scores are in-sample on 200k sampled labelled rows; CV scores are out-of-fold; student-test scores are on the 10 held-out trips.*
