"""Copy the seven user-supplied artifacts without changing bytes or overwriting data.

The source can be the original ZIP or a folder that already contains the seven files, e.g.
    python scripts/import_dataset.py "C:/Users/<you>/Downloads/outputs (1)"
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def read_source(source, expected):
    """Return {name: bytes} for the seven audited artifacts from a ZIP or a folder."""
    if source.is_dir():
        missing=[name for name in expected if not (source/name).is_file()]
        if missing: raise SystemExit(f'Folder is missing audited artifacts: {missing}')
        files={name:(source/name).read_bytes() for name in expected}
    else:
        with zipfile.ZipFile(source) as archive:
            if set(archive.namelist()) != set(expected): raise SystemExit('Archive must contain exactly the seven audited artifacts')
            for name in expected:
                if archive.getinfo(name).file_size > 100*1024*1024: raise SystemExit('Artifact exceeds audited size limit')
            files={name:archive.read(name) for name in expected}
    return files

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path,help='original ZIP file or folder containing the seven artifacts')
    parser.add_argument('--destination',type=Path,default=ROOT/'data/authoritative')
    args=parser.parse_args()
    expected=json.loads((ROOT/'docs/AUTHORITATIVE_CHECKSUMS.json').read_text())
    files=read_source(args.source,expected)
    # Verify everything before writing any data.
    for name,digest in expected.items():
        if hashlib.sha256(files[name]).hexdigest()!=digest: raise SystemExit(f'Artifact differs from audited input: {name}')
        out=args.destination/name
        if out.exists() and out.read_bytes()!=files[name]: raise SystemExit(f'Refusing to overwrite differing artifact: {name}')
    args.destination.mkdir(parents=True,exist_ok=True)
    for name in expected:
        out=args.destination/name
        if not out.exists(): out.write_bytes(files[name])
    print(f'Verified seven byte-identical artifacts in {args.destination}')

if __name__=='__main__':main()
