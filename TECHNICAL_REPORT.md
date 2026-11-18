# PathSnap Technical Report: Distributed Map-Matching & Sequential Route Reconstruction
**Institution:** PES University, Department of Computer Science and Engineering, RR Campus  
**Course:** UE24CS352A - Machine Learning Hackathon  
**Deliverable:** Technical Report & Model Evaluation Document  

---

## 2.2 Dataset Understanding and Exploratory Data Analysis (EDA)

### 1. Data Preprocessing & Pipeline Architecture
The PathSnap raw dataset consists of distinct partitions: 80 training trips, 10 student-testing trips, and 10 reserved evaluator-testing trips. Each trip contains 4 file formats:
- `.track`: Raw observed vehicle trajectories containing longitude, latitude, and timestamps (nominally 1 Hz).
- `.nodes`: Road network junction vertices with angular coordinates (0-indexed).
- `.arcs`: Directed road network edges with `(from_node, to_node)` pairs (0-indexed `arc_id`).
- `.route`: Ground-truth traversed road segments forming a continuous topological sequence.

#### Preprocessing & Coordinate Projection
Because raw coordinates are provided in degrees (WGS84), Euclidean calculations in latitude/longitude introduce severe spherical distortions. We implemented a local azimuthal geodesic projection centered at the mean trajectory latitude:
$$\text{scale}_x = 111,139 \times \cos(\phi_{\text{mean}}), \quad \text{scale}_y = 111,139$$
All GPS observations, road node vertices, and segment interpolation points were projected into physical metric coordinates $(x_m, y_m)$ in meters.

#### Candidate Retrieval & Spatial Indexing
To match GPS points against tens of thousands of road segments without quadratic complexity:
- Each road segment is interpolated with 5 points $(t \in \{0.0, 0.25, 0.5, 0.75, 1.0\})$.
- A spatial $k$-d tree is constructed over the discretized segments.
- Querying the $k$-d tree yielded **98.63% Candidate Recall** (Top-15 candidates per point) with a **Candidate Miss Rate of only 1.37%**.

### 2. Feature Engineering Rationale
For every candidate arc at observation $t$, we engineered 10 geometric, kinematic, and topological features:
1. `dist_to_segment`: Orthogonal distance from GPS point to candidate road segment (meters).
2. `log_dist`: $\log(1 + d)$ to compress outlier spikes caused by urban canyon GPS noise.
3. `heading_diff`: Absolute angular divergence $|\theta_{\text{vehicle}} - \theta_{\text{segment}}| \in [0, \pi]$ between instantaneous vehicle trajectory heading and segment orientation.
4. `cos_heading_diff`: $\cos(\Delta \theta)$, providing a smooth, non-linear similarity measure.
5. `proj_ratio_t`: Parametric projection ratio $t \in [0, 1]$ indicating position along the segment.
6. `segment_length`: Physical length of the candidate road arc.
7. `gps_speed`: Estimated instantaneous speed $\sqrt{\Delta x^2 + \Delta y^2} / \Delta t$.
8. `time_gap`: Sampling interval $\Delta t$ in seconds.
9. `dist_rank`: Discrete ranking of candidate arcs sorted by distance.
10. `is_closest`: Binary flag indicating the spatial nearest neighbor.

### EDA Findings & Visualizations
- **Trajectory Lengths:** Training trips average $2,605$ observations (ranging from $128$ to $12,987$ points). Student-test trips average $1,916$ points.
- **Sampling Interval Regularity:** Over $98.2\%$ of consecutive points are sampled at exactly $1.0\text{ s}$ intervals.
- **Missing Values:** Zero missing or null entries detected across `.track` and `.nodes`.
- **Folium Map:** Interactive maps generated for test trajectories showing raw GPS dots, underlying road graphs, and ground-truth polyline alignments (`maps/trip_00000007_reconstructed.html`).

---

## 3. Model Experimentation and Comparison

A strict multi-family experimental benchmark was conducted on 5 diverse algorithm classes using **5-Fold GroupKFold Cross-Validation** (grouped by `trip_id` to prevent data leakage):

### 3.1 Overall Model Comparison Table
| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | Boosting | `max_iter=150, lr=0.1, max_leaf_nodes=31` | **0.8231** | **79.42%** | **0.6265** | **0.8755** | **0.0925** | 0.359s |
| **Random Forest** | Ensemble | `n_estimators=100, max_depth=12` | 0.8213 | 79.39% | 0.6289 | 0.8785 | 0.0913 | 0.387s |
| **Decision Tree** | Tree-based | `max_depth=12, min_samples_leaf=20` | 0.8107 | 79.07% | 0.6252 | 0.8746 | 0.1004 | 0.018s |
| **MLP Classifier** | Neural Network | `hidden_layers=(64, 32), lr=0.001` | 0.8164 | 78.99% | 0.6294 | 0.8744 | 0.0951 | 0.147s |
| **Logistic Regression** | Linear/Probabilistic | `C=1.0, StandardScaler, max_iter=500` | 0.8050 | 77.39% | 0.6111 | 0.8367 | 0.1338 | 0.014s |

---

## 3.2 Compulsory Model Analysis

### 1. HistGradientBoosting (Chosen Candidate Model)
- **Suitability:** Fast histogram binning handles large tabular candidate datasets (~2.6M rows) seamlessly, naturally captures non-linear feature interactions (e.g. speed vs. heading deviation), and produces well-calibrated class probabilities.
- **Strengths & Limitations:** Native support for non-linear thresholds and rapid prediction; limitation is susceptibility to overfitting on sparse negative candidate samples without regularization.
- **Loss & Hyperparameters:** Binary cross-entropy (log-loss) objective; default learning rate $0.1$, max leaf nodes $31$.
- **Performance:** Student-Test Top-1 Accuracy: $79.42\%$, Macro F1: $0.8755$, Log Loss: $0.0925$.
- **Generalization:** No severe overfitting observed; CV Macro F1 ($0.8231$) generalized robustly to unseen student-test trips ($0.8755$).

### 2. Random Forest Classifier
- **Suitability:** Ensemble bagging of decorrelated decision trees reduces variance in noisy GPS environments.
- **Strengths & Limitations:** High stability across validation folds, robust to feature scale; computationally expensive training ($55.31\text{ s}$).
- **Performance:** CV Macro F1: $0.8213$, Top-1 Accuracy: $79.39\%$, Macro F1: $0.8785$, Log Loss: $0.0913$.

### 3. Decision Tree Classifier
- **Suitability:** Establishes interpretable orthogonal decision boundaries based on distance thresholds and heading differences.
- **Strengths & Limitations:** Extremely fast inference ($0.018\text{ s}$); prone to greedy sub-optimal splits and step discontinuities.
- **Performance:** CV Macro F1: $0.8107$, Top-1 Accuracy: $79.07\%$, Log Loss: $0.1004$.

### 4. Multi-Layer Perceptron (Neural Network)
- **Suitability:** Non-linear representation learning via stacked dense layers with ReLU activations.
- **Strengths & Limitations:** Capable of smooth decision manifolds; requires feature standardization and exhibits higher convergence variance.
- **Performance:** CV Macro F1: $0.8164$, Top-1 Accuracy: $78.99\%$, Log Loss: $0.0951$.

### 5. Logistic Regression
- **Suitability:** Linear probabilistic baseline establishing the linear separability of candidate features.
- **Strengths & Limitations:** Hyper-fast inference ($0.014\text{ s}$); cannot model non-linear interactions without polynomial expansion.
- **Performance:** CV Macro F1: $0.8050$, Top-1 Accuracy: $77.39\%$, Log Loss: $0.1338$.

---

## 4. Bayesian Hyperparameter Optimisation

Bayesian Hyperparameter Optimisation was executed using **Optuna's Tree-structured Parzen Estimator (TPE)** over 15 trials using 3-Fold Group Cross-Validation on training data only:
- **Search Space:**
  - `learning_rate` $\in [0.03, 0.25]$ (log-uniform)
  - `max_iter` $\in [100, 250]$ (step: 25)
  - `max_leaf_nodes` $\in [20, 63]$
  - `min_samples_leaf` $\in [10, 100]$ (step: 10)
  - `l2_regularization` $\in [10^{-4}, 10.0]$ (log-uniform)
- **Objective:** Maximize Group-aware Validation Macro F1-score.
- **Optimal Hyperparameters Obtained:**
  - `learning_rate`: $0.0443$
  - `max_iter`: $150$
  - `max_leaf_nodes`: $43$
  - `min_samples_leaf`: $50$
  - `l2_regularization`: $0.00286$

### Tuned vs. Untuned Comparison
| Configuration | CV Macro F1 | Student-Test Top-1 Accuracy | Student-Test Macro F1 | Log Loss | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Untuned Baseline | 0.8231 | 79.42% | 0.8755 | 0.0925 | 0.359s |
| **Bayesian Tuned** | **0.9078** | **79.69%** | **0.8774** | **0.0910** | 0.646s |

---

## 5. HMM and Viterbi Route Reconstruction

Independent point-wise classification frequently outputs disconnected or zigzagging road segments. We resolved this through a Hidden Markov Model (HMM) solved via dynamic programming (Viterbi decoding):
1. **Hidden States:** Candidate road segments $C_t = \{c_{t,1}, \dots, c_{t,k}\}$ at observation epoch $t$.
2. **Emission Probabilities:** Modeled by combining model predictive probability with geodesic distance:
   $$\log P(z_t | c_t) = \log(\hat{p}_{\text{model}}) - \frac{d(z_t, c_t)^2}{2\sigma^2}$$
3. **Transition Probabilities:** Enforces network topology:
   - $P(c_t | c_{t-1})$ is maximized when $c_t = c_{t-1}$ (vehicle stays on same arc) or $v(c_{t-1}) = u(c_t)$ (continuous transition across adjacent junction).
   - Non-adjacent jumps incur an exponential topological distance penalty: $\exp(-\Delta / \beta)$.
4. **Handling Disconnects & Gaps:** Disconnected transitions are penalized dynamically proportional to spatial Euclidean gap distance.

### Experimental Impact of HMM/Viterbi Decoding
| Metric | Classifier-Only | Classifier + HMM Viterbi | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Top-1 Accuracy** | 81.63% | **85.45%** | **+3.82%** |
| **Mean Route Continuity** | 88.82% | **98.50%** | **+9.68%** |
| **Mean Sequence Jaccard** | 68.73% | **79.61%** | **+10.88%** |

---

## 6. Deployment Architecture & System Demonstration

### Two-Laptop Distributed System:
1. **Laptop 1 (FastAPI Model & Evaluation Engine):**
   - Serves REST endpoints (`/trips`, `/predict_and_evaluate`).
   - Ground-truth evaluation occurs strictly server-side, preventing evaluation label leakage.
2. **Laptop 2 (Evaluator Web UI Dashboard):**
   - Dynamic selection across all 10 evaluation trips.
   - Interactive dual polyline overlay: Green = Ground Truth, Orange Dashed = Reconstructed HMM Route, Blue = Raw GPS.
   - Instant readout of Accuracy, Continuity, Sequence Jaccard, and Latency.
