#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${AICAFE_PYTHON:-.venv/bin/python}"
"$PYTHON_BIN" -m unittest discover -s tests -v
node tests/test_connection_ui.cjs
node --check web/app.js
node --check web/bundle.js
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
from fastapi.testclient import TestClient
from app.server import app
root = Path.cwd()
manifest = json.loads((root / 'samples/manifest.json').read_text())
assert len(manifest['clips']) == 64
for clip in manifest['clips']:
    assert (root / clip['file']).is_file(), clip['id']
with TestClient(app) as client:
    assert client.get('/api/status').status_code == 200
    assert len(client.get('/api/clips').json()['clips']) >= 64
    assert client.get('/web/index.html').status_code == 200
print('Cloud smoke check passed: server, UI and 64 sample files. No VLM inference performed.')
PY
