"""
Bayesian Hyperparameter Optimisation for HistGradientBoosting
Using Optuna with Group-aware Cross-Validation on the 80 training trips.
Records search history, objective evaluations, best parameters, and compares untuned vs tuned.
"""
import os
import time
import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

CACHE_DIR = r"C:\Users\admin\.gemini\antigravity\scratch\PathSnap\data_cache"
MODELS_DIR = r"C:\Users\admin\.gemini\antigravity\scratch\PathSnap\models"
PLOTS_DIR = r"C:\Users\admin\.gemini\antigravity\scratch\PathSnap\model_plots"

def run_bayesian_optimization():
    print("Loading datasets for Bayesian Hyperparameter Optimisation...")
    train_data = np.load(os.path.join(CACHE_DIR, "train_data.npz"))
    X_train, y_train, groups_train = train_data['X'], train_data['y'], train_data['groups']
    
    test_data = np.load(os.path.join(CACHE_DIR, "student_test_data.npz"))
    X_test, y_test = test_data['X'], test_data['y']
    meta_test = pd.read_csv(os.path.join(CACHE_DIR, "student_test_meta.csv"))
    meta_test['is_correct'] = y_test
    
    # Stratified/balanced fit subset for rapid, rigorous evaluation
    pos_mask = y_train == 1
    neg_indices = np.where(~pos_mask)[0]
    np.random.seed(42)
    selected_neg = np.random.choice(neg_indices, size=int(np.sum(pos_mask) * 4), replace=False)
    fit_indices = np.concatenate([np.where(pos_mask)[0], selected_neg])
    np.random.shuffle(fit_indices)
    
    X_fit = X_train[fit_indices]
    y_fit = y_train[fit_indices]
    groups_fit = groups_train[fit_indices]
    
    # 3-Fold Group CV for Optuna trials (using training data only, strictly preventing evaluator leakage)
    gkf = GroupKFold(n_splits=3)
    
    # Track trial histories
    trial_records = []
    
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
        max_iter = trial.suggest_int("max_iter", 100, 250, step=25)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 20, 63)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 100, step=10)
        l2_regularization = trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True)
        
        cv_scores = []
        for train_idx, val_idx in gkf.split(X_fit, y_fit, groups_fit):
            clf = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=42
            )
            clf.fit(X_fit[train_idx], y_fit[train_idx])
            preds = clf.predict(X_fit[val_idx])
            f1 = f1_score(y_fit[val_idx], preds, average='macro')
            cv_scores.append(f1)
            
        mean_f1 = float(np.mean(cv_scores))
        trial_records.append({
            'trial_number': trial.number,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'max_leaf_nodes': max_leaf_nodes,
            'min_samples_leaf': min_samples_leaf,
            'l2_regularization': l2_regularization,
            'cv_macro_f1': mean_f1
        })
        return mean_f1
        
    print("Starting Bayesian search with 15 trials...")
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=15)
    opt_time = time.time() - t0
    
    print(f"Optimization finished in {opt_time:.1f}s")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best CV Macro F1: {study.best_value:.4f}")
    print("Best Hyperparameters:", study.best_params)
    
    # Save search log CSV
    df_trials = pd.DataFrame(trial_records)
    trials_csv = os.path.join(CACHE_DIR, "bayesian_trials_log.csv")
    df_trials.to_csv(trials_csv, index=False)
    print(f"Saved trials log to {trials_csv}")
    
    # Train final tuned model with best params
    print("\nRefitting final tuned model on full fit set...")
    best_params = study.best_params
    tuned_model = HistGradientBoostingClassifier(
        learning_rate=best_params['learning_rate'],
        max_iter=best_params['max_iter'],
        max_leaf_nodes=best_params['max_leaf_nodes'],
        min_samples_leaf=best_params['min_samples_leaf'],
        l2_regularization=best_params['l2_regularization'],
        random_state=42
    )
    tuned_model.fit(X_fit, y_fit)
    joblib.dump(tuned_model, os.path.join(MODELS_DIR, "best_tuned_histgradientboosting.joblib"))
    
    # Evaluate Tuned Model on Student Test
    t_start_inf = time.time()
    test_probs = tuned_model.predict_proba(X_test)[:, 1]
    t_inf_tuned = time.time() - t_start_inf
    test_preds = (test_probs >= 0.5).astype(int)
    
    test_macro_f1 = float(f1_score(y_test, test_preds, average='macro'))
    test_loss = float(log_loss(y_test, test_probs))
    
    # Observation Top-1 Accuracy
    meta_eval = meta_test.copy()
    meta_eval['prob'] = test_probs
    idx_max = meta_eval.groupby(['trip_id', 'obs_idx'])['prob'].idxmax()
    best_cand = meta_eval.loc[idx_max].sort_values(['trip_id', 'obs_idx'])
    top1_acc = float(np.mean(best_cand['is_correct']))
    
    # Load untuned model results
    untuned_res = pd.read_csv(os.path.join(CACHE_DIR, "model_comparison_table.csv"))
    untuned_row = untuned_res[untuned_res['Model'] == 'HistGradientBoosting'].iloc[0]
    
    comparison = pd.DataFrame([
        {
            "Configuration": "Untuned Baseline (Default)",
            "CV Macro F1-score": untuned_row['CV Macro F1-score'],
            "Student-Test Top-1 Accuracy": untuned_row['Top-1 Accuracy'],
            "Student-Test Macro F1-score": untuned_row['Student-Test Macro F1-score'],
            "Log Loss": untuned_row['Log Loss'],
            "Inference Time": untuned_row['Inference Time']
        },
        {
            "Configuration": "Bayesian Tuned (TPE)",
            "CV Macro F1-score": round(study.best_value, 4),
            "Student-Test Top-1 Accuracy": f"{top1_acc*100:.2f}%",
            "Student-Test Macro F1-score": round(test_macro_f1, 4),
            "Log Loss": round(test_loss, 4),
            "Inference Time": f"{t_inf_tuned:.3f}s"
        }
    ])
    
    comp_csv = os.path.join(CACHE_DIR, "tuned_vs_untuned_comparison.csv")
    comparison.to_csv(comp_csv, index=False)
    print("\n--- TUNED VS UNTUNED COMPARISON ---")
    print(comparison)
    
    # Plot Bayesian Optimization Trials Convergence
    plt.figure(figsize=(9, 4.5))
    plt.plot(df_trials['trial_number'], df_trials['cv_macro_f1'], 'o-', color='#8e44ad', linewidth=2, markersize=7)
    plt.axhline(study.best_value, color='#27ae60', linestyle='--', label=f'Best CV F1: {study.best_value:.4f}')
    plt.title("Bayesian Hyperparameter Optimisation Convergence (Optuna TPE)")
    plt.xlabel("Trial Number")
    plt.ylabel("Validation Macro F1-score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bayesian_opt_convergence.png"), dpi=300)
    plt.close()
    
    print("Optimization convergence plot saved to", os.path.join(PLOTS_DIR, "bayesian_opt_convergence.png"))

if __name__ == "__main__":
    run_bayesian_optimization()
