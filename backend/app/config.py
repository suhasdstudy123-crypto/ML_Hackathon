from dataclasses import dataclass, field
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    trip_dir: Path = field(default_factory=lambda: Path(os.getenv('PATHSNAP_TRIP_DIR', ROOT / 'evaluator_test/evaluator_test')))
    feature_dir: Path = field(default_factory=lambda: Path(os.getenv('PATHSNAP_FEATURE_DIR', ROOT / 'data/authoritative')))
    manifest_path: Path = field(default_factory=lambda: Path(os.getenv('PATHSNAP_MODEL_MANIFEST', ROOT / 'models/model_manifest.json')))
    notebook_demo_mode: bool = field(default_factory=lambda: os.getenv('PATHSNAP_NOTEBOOK_DEMO_MODE', '1') == '1')
    parity_report: Path = field(default_factory=lambda: Path(os.getenv('PATHSNAP_PARITY_REPORT', ROOT / 'docs/NOTEBOOK_PARITY.json')))
    origins: tuple = field(default_factory=lambda: tuple(x.strip() for x in os.getenv('PATHSNAP_CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',') if x.strip()))
    max_workers: int = field(default_factory=lambda: int(os.getenv('PATHSNAP_MAX_WORKERS', '1')))
    max_jobs: int = field(default_factory=lambda: int(os.getenv('PATHSNAP_MAX_JOBS', '16')))
    retention_seconds: int = field(default_factory=lambda: int(os.getenv('PATHSNAP_JOB_RETENTION_SECONDS', '3600')))
    timeout_seconds: float = field(default_factory=lambda: float(os.getenv('PATHSNAP_JOB_TIMEOUT_SECONDS', '120')))
    max_observations: int = 25000
    max_arcs: int = 400000
    max_file_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        if '*' in self.origins or 'null' in self.origins:
            raise ValueError('CORS requires explicit HTTP frontend origins')
        if not 1 <= self.max_workers <= 4 or not self.max_workers <= self.max_jobs <= 128:
            raise ValueError('Invalid job capacity')
        if self.retention_seconds <= 0 or self.timeout_seconds <= 0:
            raise ValueError('Job timing limits must be positive')
