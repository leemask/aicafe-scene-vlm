"""Validate a completed paired run and build a side-by-side evidence review page."""
import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_vlm import MODES, sha256


def safe_path(folder, relative):
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("Evidence path leaves run folder")
    return path


def build_report(folder):
    run = json.loads((folder / "run.json").read_text())
    if run["status"] != "completed":
        raise ValueError("Run is not completed")
    manifest = json.loads((ROOT / "samples/manifest.json").read_text())
    clips = {c["id"]: c for c in manifest["clips"]}
    if len(set(run["clip_ids"])) != len(run["clip_ids"]):
        raise ValueError("Duplicate clips")
    cards, pairs, counts = [], [], {mode: 0 for mode in MODES}
    for clip_id in run["clip_ids"]:
        panels, results = [], {}
        clip = clips[clip_id]
        video_sha = sha256(ROOT / clip["file"])
        for mode, budget in MODES.items():
            record = json.loads(safe_path(folder, f"{mode}/{clip_id}.json").read_text())
            assert record["clip_id"] == clip_id and record["mode"] == mode
            assert record["input_video_sha256"] == video_sha
            times = record["frame_times"]
            assert len(times) == budget and times == sorted(set(times))
            assert all(0 <= t < clip["duration"] for t in times)
            assert len(record["evidence_frames"]) == len(times)
            frames = []
            for t, evidence in zip(times, record["evidence_frames"]):
                path = safe_path(folder, evidence["file"])
                assert evidence["time"] == t and sha256(path) == evidence["sha256"]
                frames.append('<figure><img loading="lazy" src="' + html.escape(evidence["file"], quote=True)
                              + '"><figcaption>' + f'{t:.3f}s' + '</figcaption></figure>')
            counts[mode] += int(record["parse_ok"])
            results[mode] = record
            panels.append('<section><h3>' + mode + '</h3><p>' + html.escape(record["summary_ko"])
                          + '</p><p>모델 분류: ' + html.escape(record["category"])
                          + '</p><div class="frames">' + ''.join(frames)
                          + '</div><details><summary>원문·근거·한계</summary><pre>'
                          + html.escape(record["raw"]) + '</pre></details></section>')
        pairs.append({"clip_id": clip_id, "category_changed": results["uniform3"]["category"] != results["uniform6"]["category"]})
        cards.append('<article><h2>' + html.escape(clip_id) + '</h2><div class="pair">'
                     + ''.join(panels) + '</div></article>')
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLM 3장·6장 근거 비교</title><style>body{font:16px/1.6 system-ui;margin:24px;background:#f5f7fa;color:#162130}article{background:white;padding:20px;margin:20px 0;border-radius:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}.frames{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}figure{margin:0}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{border-bottom:1px solid #ddd}@media(max-width:760px){.pair{grid-template-columns:1fr}}</style>
<h1>균등 3장·6장 VLM 출력 비교</h1><p>모든 출력은 미검토 모델 추정입니다. 설명 변화와 JSON 형식 성공은 정확도 개선이 아닙니다. 아래 이미지는 모델에 넣기 전 최대 576×324로 축소한 입력을 PNG로 저장한 것입니다. 전체 영상의 연속 관찰과 다릅니다.</p>'''
    (folder / "review.html").write_text(document + ''.join(cards) + '</html>', encoding="utf-8")
    integrity = {"clips": len(pairs), "inferences": 2 * len(pairs), "valid_json": counts,
                 "category_changed": sum(p["category_changed"] for p in pairs),
                 "video_and_evidence_sha256": "passed", "timestamp_budgets": "passed",
                 "human_review": "not_performed", "accuracy_estimated": False}
    (folder / "integrity.json").write_text(json.dumps(integrity, indent=2) + '\n')
    return integrity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        parser.error("Invalid run ID")
    print(json.dumps(build_report(ROOT / "experiments" / args.run_id), indent=2))
