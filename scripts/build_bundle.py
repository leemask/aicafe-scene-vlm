"""Make a self-contained offline data bundle from persisted, real model jobs."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'samples/manifest.json').read_text(encoding='utf-8'))
jobs = [json.loads(p.read_text(encoding='utf-8')) for p in (root / 'results').glob('*.json')]
jobs = sorted([j for j in jobs if 'clip_id' in j], key=lambda j: j.get('created_at', ''), reverse=True)
(root / 'web/bundle.js').write_text('window.SCENE_BUNDLE = ' + json.dumps({'clips': manifest['clips'], 'jobs': jobs}, ensure_ascii=False) + ';', encoding='utf-8')
print(f'Bundled {len(manifest["clips"])} videos and {len(jobs)} jobs')
