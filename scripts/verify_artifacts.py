"""Verify preserved historical artifacts and, when present, supplied authoritative data."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
checks=json.loads((ROOT/'docs/ARTIFACT_CHECKSUMS.json').read_text())
for p,digest in checks.items():
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
print(f'{len(checks)} historical artifacts unchanged')
path=ROOT/'data/authoritative'
if path.is_dir():
    checks=json.loads((ROOT/'docs/AUTHORITATIVE_CHECKSUMS.json').read_text())
    for p,digest in checks.items():assert hashlib.sha256((path/p).read_bytes()).hexdigest()==digest,p
    print(f'{len(checks)} authoritative artifacts unchanged')
else:print('Authoritative dataset not installed locally')
