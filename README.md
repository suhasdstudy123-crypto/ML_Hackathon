# PathSnap: Distributed Map-Matching & Sequential Route Reconstruction
**PES University - Department of Computer Science and Engineering**  
**Course:** UE24CS352A Machine Learning  

---

## 📌 Project Overview
PathSnap is an end-to-end distributed machine learning system engineered to reconstruct accurate vehicle routes from noisy GPS trajectories matched against a complex geographical road network graph.

### Key Architectural Highlights:
1. **Candidate Retrieval & Spatial Indexing:** $k$-d tree with 5-point segment interpolation providing **98.63% candidate recall** with low miss rate ($1.37\%$).
2. **Feature Engineering:** 10 geometric, kinematic, and topological features (including geodesic orthogonal projection, angular heading divergence, speed, and time delta).
3. **5 Model Family Benchmarking:** Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, and MLP neural networks evaluated using 5-Fold **GroupKFold** across trips to strictly prevent data leakage.
4. **Bayesian Hyperparameter Optimisation:** Optuna TPE tuning over learning rate, tree depth, leaf limits, and L2 regularization, boosting cross-validation Macro F1 to **0.9078**.
5. **HMM & Viterbi Decoding:** Emission probability derived from calibrated model score + spatial distance; topological transition model penalizing network disconnects. Achieved **85.45% Top-1 Accuracy**, **98.50% Route Continuity**, and **79.61% Sequence Jaccard**.
6. **Two-Laptop Distributed Deployment:**
   - **Laptop 1:** FastAPI backend hosting model weights, candidate pipeline, and server-side ground-truth evaluation.
   - **Laptop 2:** Interactive Leaflet / Folium dashboard displaying the 10 evaluation trips, comparative trajectory overlays, and live performance metrics.

---

## 📂 Repository Structure
```
PathSnap/
├── README.md                           # Documentation & execution instructions
├── requirements.txt                    # Project dependencies
├── build_cache.py                      # Preprocessing and dataset caching script
├── feature_engineering.py              # Spatial indexing, candidate sampling & feature extractor
├── train_and_evaluate.py               # 5-model benchmarking framework & evaluation metrics
├── bayesian_tuning.py                  # Optuna Bayesian hyperparameter optimisation
├── hmm_viterbi.py                      # Hidden Markov Model & Viterbi path decoder
├── generate_maps.py                    # Folium interactive map visualisations
├── api_server.py                       # FastAPI backend server (Laptop 1)
├── evaluator_ui/                       # Evaluator Frontend interface (Laptop 2)
│   └── index.html
├── data_cache/                         # Preprocessed datasets and summary CSV tables
│   ├── model_comparison_table.csv
│   ├── tuned_vs_untuned_comparison.csv
│   └── hmm_summary_comparison.csv
├── models/                             # Saved serialized model artifacts
│   ├── best_tuned_histgradientboosting.joblib
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   └── random_forest.joblib
├── model_plots/                        # Evaluation figures and curves
│   ├── macro_f1_comparison.png
│   ├── top1_accuracy_comparison.png
│   ├── log_loss_comparison.png
│   └── bayesian_opt_convergence.png
└── maps/                               # Interactive Folium map HTML files
    ├── trip_00000000_reconstructed.html
    └── trip_00000007_reconstructed.html
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd PathSnap
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Two-Laptop Deployment Execution

### Laptop 1 (Model and API Server)
Run the FastAPI backend on the local network:
```bash
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```
- Note down Laptop 1's local IP address (e.g. `192.168.1.45`) using `ipconfig`.
- Verify the server is running by opening `http://localhost:8000/docs` in your browser.

### Laptop 2 (Evaluator Dashboard UI)
1. Open `evaluator_ui/index.html` in any modern web browser.
2. In the **API Endpoint URL** field, enter Laptop 1's IP address (e.g., `http://192.168.1.45:8000`).
3. Click **Test Connection** — the indicator will turn green.
4. Select any of the 10 reserved evaluation trips from the dropdown.
5. Click **Run Evaluation** to view the live matched trajectory, reconstructed route, and performance metrics.

---

## 📊 Summary of Experimental Results

### 1. Overall Model Comparison (5 Algorithm Families)
| Model | Algorithm Family | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | Boosting | **0.8231** | **79.42%** | **0.6265** | **0.8755** | **0.0925** | 0.359s |
| Random Forest | Ensemble | 0.8213 | 79.39% | 0.6289 | 0.8785 | 0.0913 | 0.387s |
| Decision Tree | Tree-based | 0.8107 | 79.07% | 0.6252 | 0.8746 | 0.1004 | 0.018s |
| MLP | Neural Network | 0.8164 | 78.99% | 0.6294 | 0.8744 | 0.0951 | 0.147s |
| Logistic Regression | Linear/Probabilistic | 0.8050 | 77.39% | 0.6111 | 0.8367 | 0.1338 | 0.014s |

### 2. Bayesian Tuning (Untuned vs Tuned)
| Configuration | CV Macro F1 | Student-Test Top-1 Accuracy | Student-Test Macro F1 | Log Loss | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Untuned Baseline (Default) | 0.8231 | 79.42% | 0.8755 | 0.0925 | 0.359s |
| **Bayesian Tuned (TPE Optuna)** | **0.9078** | **79.69%** | **0.8774** | **0.0910** | 0.646s |

### 3. Route Reconstruction (Classifier vs Classifier + HMM Viterbi)
| Method | Mean Top-1 Accuracy | Mean Route Continuity | Mean Sequence Jaccard |
| :--- | :---: | :---: | :---: |
| Classifier-Only Predictions | 81.63% | 88.82% | 68.73% |
| **Classifier + HMM/Viterbi Decoding** | **85.45%** | **98.50%** | **79.61%** |
