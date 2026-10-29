"""Download exact source videos and copy bundled partial annotations for reproduction."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--research', type=Path, required=True)
args = p.parse_args()
args.research.mkdir(parents=True, exist_ok=True)
(args.research / 'raw').mkdir(exist_ok=True)
(args.research / 'annotations').mkdir(exist_ok=True)
clips = json.loads((root / 'samples/manifest.json').read_text(encoding='utf-8'))['clips']
sources = {c['source_name']: c for c in clips}
records = []
for name, clip in sources.items():
    dest = args.research / 'raw' / (name + '.avi')
    if not dest.exists():
        temp = dest.with_suffix('.part')
        with urllib.request.urlopen(clip['source_url'], timeout=120) as response, temp.open('wb') as out:
            shutil.copyfileobj(response, out, 1024 * 1024)
        if hashlib.file_digest(temp.open('rb'), 'sha256').hexdigest() != clip['source_sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')
        temp.replace(dest)
    elif hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest() != clip['source_sha256']:
        raise ValueError(f'Existing source checksum mismatch: {name}')
    annotation = name + '.activities.yml'
    shutil.copyfile(root / 'samples/annotations' / annotation, args.research / 'annotations' / annotation)
    records.append({'name': name, 'url': clip['source_url'], 'path': str(dest)})
    print(name, dest.stat().st_size, flush=True)
(args.research / 'sources.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
