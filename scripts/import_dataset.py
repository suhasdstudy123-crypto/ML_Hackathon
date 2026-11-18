"""Copy the seven user-supplied artifacts without changing bytes or overwriting data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zip_path',type=Path)
    parser.add_argument('--destination',type=Path,default=ROOT/'data/authoritative')
    args=parser.parse_args()
    expected=json.loads((ROOT/'docs/AUTHORITATIVE_CHECKSUMS.json').read_text())
    with zipfile.ZipFile(args.zip_path) as archive:
        if set(archive.namelist()) != set(expected): raise SystemExit('Archive must contain exactly the seven audited artifacts')
        # Verify entire archive before writing any data.
        for name,digest in expected.items():
            info=archive.getinfo(name)
            if info.file_size > 100*1024*1024: raise SystemExit('Artifact exceeds audited size limit')
            value=archive.read(name)
            if hashlib.sha256(value).hexdigest()!=digest: raise SystemExit(f'Artifact differs from audited input: {name}')
            out=args.destination/name
            if out.exists() and out.read_bytes()!=value: raise SystemExit(f'Refusing to overwrite differing artifact: {name}')
        args.destination.mkdir(parents=True,exist_ok=True)
        for name in expected:
            out=args.destination/name
            if not out.exists(): out.write_bytes(archive.read(name))
    print(f'Verified seven byte-identical artifacts in {args.destination}')

if __name__=='__main__':main()
