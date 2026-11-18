"""Export the trained LightGBM pipeline into backend-ready, version-independent artifacts.

Input : lightgbm/models/lightgbm.joblib  (SimpleImputer(median) -> StandardScaler -> LGBMClassifier)
        lightgbm/models/lightgbm_booster.txt (native booster saved by the training notebook)
Output: lightgbm/models/lightgbm_preprocessing.json  (exact fitted imputer/scaler parameters)
        lightgbm/models/model_manifest.json          (backend model contract with SHA-256 hashes)

Nothing is retrained. Run once after training:  python scripts/export_lightgbm_artifacts.py
"""
import hashlib
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.models.contract import FEATURES, FEATURE_TYPES  # noqa: E402

MODEL_DIR = ROOT / 'lightgbm/models'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import joblib
    import lightgbm
    import sklearn
    with warnings.catch_warnings():
        # The pipeline was pickled with a newer sklearn; we only read fitted numeric arrays from it.
        warnings.simplefilter('ignore')
        pipe = joblib.load(MODEL_DIR / 'lightgbm.joblib')
    names = [name for name, _ in pipe.steps]
    if names != ['impute', 'scale', 'model']:
        raise SystemExit(f'Unexpected pipeline steps: {names}')
    imputer, scaler, model = pipe.named_steps['impute'], pipe.named_steps['scale'], pipe.named_steps['model']
    if imputer.strategy != 'median' or not (scaler.with_mean and scaler.with_std):
        raise SystemExit('Pipeline preprocessing is not median imputation + standardisation')

    # The native booster file must be exactly the booster inside the pipeline.
    booster_path = MODEL_DIR / 'lightgbm_booster.txt'
    if model.booster_.model_to_string() != booster_path.read_text():
        raise SystemExit('lightgbm_booster.txt differs from the booster inside lightgbm.joblib')

    # repr() of a Python float round-trips exactly, so JSON keeps every bit.
    pre = {
        'type': 'median_impute_then_standardize',
        'source': 'lightgbm/models/lightgbm.joblib steps impute + scale (fitted on all labelled training rows)',
        'feature_names': FEATURES,
        'impute_median': [float(x) for x in imputer.statistics_],
        'scale_mean': [float(x) for x in scaler.mean_],
        'scale_std': [float(x) for x in scaler.scale_],
    }
    pre_path = MODEL_DIR / 'lightgbm_preprocessing.json'
    pre_path.write_text(json.dumps(pre, indent=2) + '\n', newline='\n')

    results = json.loads((ROOT / 'lightgbm/results/lightgbm_results.json').read_text().replace('NaN', 'null'))
    manifest = {
        'configured': True,
        'name': 'lightgbm',
        'version': 'lightgbm-600trees-seed42',
        'artifact_format': 'lightgbm_native',
        'artifact_path': booster_path.name,
        'artifact_sha256': sha256(booster_path),
        'feature_schema_version': 'notebook-features-v2',
        'runtime_pipeline': 'authoritative-notebook-v2',
        'feature_names': FEATURES,
        'feature_dtypes': FEATURE_TYPES,
        'feature_name_binding': 'positional',
        'preprocessing_path': pre_path.name,
        'preprocessing_sha256': sha256(pre_path),
        'preprocessing_feature_names': FEATURES,
        'preprocessing_training_reference': pre['source'],
        'missing_value_policy': 'preprocessor',
        'training_configuration_reference': 'lightgbm/notebooks/pathsnap_lightgbm.ipynb; lightgbm/results/lightgbm_results.json',
        'output_type': 'binary_probability',
        'positive_class': 1,
        'calibration': {'method': 'none', 'development_validation': 'student-test log loss 0.1708 (uncalibrated)'},
        'creation_metadata': {
            'seed': results['seed'],
            'training_library_versions': {'lightgbm': '4.7.0', 'scikit-learn': '1.9.1'},
            'exported_with': {'lightgbm': lightgbm.__version__, 'scikit-learn': sklearn.__version__},
            'student_test': {k: results['summary'][k] for k in ['st_macro_f1', 'st_top1_accuracy', 'st_log_loss']},
        },
    }
    (MODEL_DIR / 'model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', newline='\n')
    print(f'Wrote {pre_path.relative_to(ROOT)} and lightgbm/models/model_manifest.json')


if __name__ == '__main__':
    main()
