"""Exact, library-version-independent copy of the model's own preprocessing.

The trained artifact (lightgbm/models/lightgbm.joblib) is an sklearn Pipeline:
    SimpleImputer(strategy='median') -> StandardScaler -> LGBMClassifier
Its fitted parameters are exported to JSON by scripts/export_lightgbm_artifacts.py.
This class repeats sklearn's float64 arithmetic step for step, so the output is
bit-identical to the pipeline without unpickling sklearn objects.

Why not feature_scaler.joblib? Its means/stds differ from the model's own by ~1e-12.
That is enough to flip tree splits and move probabilities by up to 0.165.
"""
import json
from pathlib import Path
import numpy as np


class JsonStandardizer:
    def __init__(self, feature_names, medians, means, scales):
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_features_in_ = len(feature_names)
        self.medians = np.asarray(medians, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.scales = np.asarray(scales, dtype=np.float64)
        n = self.n_features_in_
        if not (self.medians.shape == self.means.shape == self.scales.shape == (n,)):
            raise ValueError('Preprocessing parameter length does not match feature count')
        if not (np.isfinite(self.medians).all() and np.isfinite(self.means).all() and np.isfinite(self.scales).all() and (self.scales > 0).all()):
            raise ValueError('Preprocessing parameters must be finite with positive scales')

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get('type') != 'median_impute_then_standardize':
            raise ValueError('Unsupported preprocessing type')
        return cls(data['feature_names'], data['impute_median'], data['scale_mean'], data['scale_std'])

    def transform(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError('Preprocessing input has the wrong number of features')
        # SimpleImputer: replace missing values with the training medians.
        missing = np.isnan(X)
        if missing.any():
            X[missing] = np.take(self.medians, np.nonzero(missing)[1])
        # StandardScaler: same in-place order of operations as sklearn (subtract, then divide).
        X -= self.means
        X /= self.scales
        return X
