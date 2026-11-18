import hashlib
import json
import logging
from pathlib import Path
from typing import Literal
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from backend.app.core.errors import AppError
from .provider import ModelProvider
from .contract import FEATURES, FEATURE_TYPES, SCHEMA_VERSION, PIPELINE_VERSION, validate_features

log = logging.getLogger('pathsnap')

class Manifest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    configured: bool = False
    name: Literal['lightgbm'] = 'lightgbm'
    version: str = Field(min_length=1)
    artifact_format: Literal['lightgbm_native', 'joblib_lightgbm']
    artifact_path: str
    artifact_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    feature_schema_version: Literal['authoritative-parquet-v1']
    runtime_pipeline: Literal['precomputed-authoritative-v1']
    feature_names: list[str]
    feature_dtypes: dict[str, str]
    preprocessing_path: str | None = None
    preprocessing_sha256: str | None = None
    preprocessing_feature_names: list[str] | None = None
    preprocessing_training_reference: str | None = None
    missing_value_policy: Literal['reject', 'native_missing', 'preprocessor'] = 'native_missing'
    training_configuration_reference: str
    output_type: Literal['binary_probability'] = 'binary_probability'
    positive_class: Literal[1] = 1
    calibration: dict = Field(default_factory=dict)
    creation_metadata: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_contract(self):
        if self.feature_names != FEATURES or self.feature_dtypes != FEATURE_TYPES:
            raise ValueError('Manifest must use exact canonical features, types, and order')
        if self.missing_value_policy == 'preprocessor':
            if not all([self.preprocessing_path, self.preprocessing_sha256, self.preprocessing_feature_names, self.preprocessing_training_reference]):
                raise ValueError('Preprocessing requires artifact hash, order and training provenance')
            if sorted(self.preprocessing_feature_names) != sorted(FEATURES):
                raise ValueError('Preprocessing must retain all canonical features exactly once')
        elif self.preprocessing_path:
            raise ValueError('Preprocessing artifact requires preprocessor policy')
        return self

class LightGBMModelProvider(ModelProvider):
    def __init__(self, manifest_path, feature_dir):
        self.path, self.feature_dir = Path(manifest_path), Path(feature_dir)
        self.manifest = self.model = self.preprocessor = None
        self.state, self.reason = 'UNCONFIGURED', 'No trained LightGBM manifest configured'

    def _file(self, value, expected_hash):
        p = Path(value)
        if not p.is_absolute(): p = self.path.parent / p
        if not p.is_file(): raise AppError('MODEL_ARTIFACT_MISSING', 'A configured model artifact is unavailable', 503)
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected_hash:
            raise AppError('INVALID_MODEL_ARTIFACT', 'Artifact checksum does not match manifest', 503)
        return p

    def load(self):
        self.model = self.preprocessor = self.manifest = None
        self.state, self.reason = 'UNCONFIGURED', 'No trained LightGBM manifest configured'
        if not self.path.is_file(): return
        try:
            m = Manifest.model_validate_json(self.path.read_text())
            self.manifest = m
            if not m.configured: return
            p = self._file(m.artifact_path, m.artifact_sha256)
            # Optional dependency imported only when a real artifact has been configured.
            import lightgbm as lgb
            if m.artifact_format == 'lightgbm_native':
                model = lgb.Booster(model_file=str(p))
            else:
                model = joblib.load(p)  # Only trusted local operator-supplied artifacts.
                if not isinstance(model, (lgb.Booster, lgb.LGBMClassifier)):
                    raise ValueError('Artifact is not a LightGBM Booster or LGBMClassifier')
            booster = model if isinstance(model, lgb.Booster) else model.booster_
            if booster.num_feature() != len(FEATURES) or booster.feature_name() != FEATURES:
                raise ValueError('Model feature count, names or order differ from manifest')
            if not str(booster.dump_model().get('objective', '')).startswith('binary '):
                raise ValueError('Only binary logistic probability output is supported')
            if isinstance(model, lgb.LGBMClassifier) and list(model.classes_) != [0, 1]:
                raise ValueError('Binary classes must be [0, 1]')
            if m.missing_value_policy == 'preprocessor':
                pp = self._file(m.preprocessing_path, m.preprocessing_sha256)
                self.preprocessor = joblib.load(pp)
                if getattr(self.preprocessor, 'n_features_in_', None) != len(FEATURES):
                    raise ValueError('Preprocessing feature count mismatch')
                if hasattr(self.preprocessor, 'feature_names_in_') and list(self.preprocessor.feature_names_in_) != m.preprocessing_feature_names:
                    raise ValueError('Preprocessing feature order mismatch')
            self.model = model
            # Validate output on genuine permitted data, never on fabricated production rows.
            probe_path = self.feature_dir / 'student_test_features.parquet'
            if not probe_path.is_file(): probe_path = self.feature_dir / 'train_features.parquet'
            if not probe_path.is_file(): raise ValueError('Real feature data required for loading compatibility probe')
            probe = pd.read_parquet(probe_path, columns=FEATURES).iloc[:5]
            self._predict(probe)
            self.state, self.reason = 'READY', None
        except Exception as exc:
            self.model = self.preprocessor = None
            self.state = 'INVALID'
            self.reason = exc.message if isinstance(exc, AppError) else 'Model compatibility validation failed; inspect server logs'
            log.exception('model_load_failed')

    def is_ready(self): return self.state == 'READY'

    def _predict(self, features):
        m = self.manifest
        validate_features(features, allow_missing=m.missing_value_policy != 'reject')
        X = features
        if self.preprocessor is not None:
            order = m.preprocessing_feature_names
            transformed = np.asarray(self.preprocessor.transform(features[order]))
            if transformed.shape != (len(features), len(FEATURES)) or not np.isfinite(transformed).all():
                raise AppError('INVALID_PREPROCESSING_OUTPUT', 'Preprocessing must produce a finite unchanged shape')
            X = pd.DataFrame(transformed, columns=order, index=features.index)[FEATURES]
        if hasattr(self.model, 'predict_proba'):
            raw = np.asarray(self.model.predict_proba(X, num_threads=1))
            if raw.shape != (len(X), 2) or not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any() or not np.allclose(raw.sum(axis=1), 1):
                raise AppError('INVALID_MODEL_OUTPUT', 'Expected binary class probability matrix')
            result = raw[:, 1]
        else:
            result = np.asarray(self.model.predict(X, raw_score=False, validate_features=True, num_threads=1))
        if result.shape != (len(X),) or not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
            raise AppError('INVALID_MODEL_OUTPUT', 'Expected finite binary candidate probabilities')
        return result

    def predict_candidate_scores(self, features):
        if not self.is_ready(): raise AppError('MODEL_NOT_READY', 'LightGBM model not connected', 503)
        try: return self._predict(features)
        except AppError: raise
        except Exception as exc:
            raise AppError('MODEL_PREDICTION_FAILED', 'Candidate scoring failed', 500) from exc

    def get_metadata(self):
        return {'name': 'lightgbm', 'version': self.manifest.version if self.manifest else None,
                'ready': self.is_ready(), 'state': self.state, 'reason': self.reason,
                'feature_schema_version': SCHEMA_VERSION, 'runtime_pipeline': PIPELINE_VERSION}
