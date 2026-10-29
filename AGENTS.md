# AICAFE Scene VLM development

Read README.md, CLOUD_HANDOFF.md, ANALYSIS_REPORT.md and VERIFICATION.md before changing behavior.

## Purpose
MIT-licensed scene understanding prototype for people moving/entering, object handling and vehicle loading. Use provided MEVA samples; this is not a validated factory alarm system.

## Development
- Python 3.11/3.12 and Node.js are required. Run `bash scripts/setup_cloud.sh` for first setup and `bash scripts/check_cloud.sh` for validation.
- Run `.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8000` for a local preview. Do not expose this unauthenticated development server publicly.
- Basic checks need no VLM weights, API key or GPU. Model installation and inference are separate, documented in README.md.
- Keep code credentials out of git. Do not commit private user uploads, caches or model weights.
- Preserve existing `results/qwen-*` baselines and evidence frames. Write comparisons under a new run identifier; do not delete baselines to trigger reruns.
- Rebuild `web/bundle.js` with `python scripts/build_bundle.py` after intentionally changing the distributed catalog or results.

## Evaluation
- Never send reference labels, filenames or camera metadata to the model.
- Partial public annotations are not exhaustive negative labels. Split by source/camera groups to avoid leakage.
- JSON validity is not accuracy; model-assigned categories are not confirmed events.
- Assistant frame review is not human ground truth. Preserve model raw outputs and clearly distinguish manual labels.
- Measure false claims in empty scenes, short-event coverage, latency and memory. Report sampled input coverage separately from action recognition.
- Do not claim GPU or live CCTV support unless implemented and tested. Code is MIT; MEVA media and annotations retain CC BY 4.0 attribution.
